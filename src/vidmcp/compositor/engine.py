"""Non-destructive layer compositor — frame graph renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from vidmcp.compositor.alpha import add_blend, load_mask_for_frame, over, screen_blend
from vidmcp.compositor.ffmpeg_ops import mux_audio
from vidmcp.core.workspace import ProjectStore
from vidmcp.effects.base import EffectContext
from vidmcp.effects.registry import get_effect_registry
from vidmcp.models.layers import BlendMode, Layer, LayerKind
from vidmcp.models.project import ProjectStatus
from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video

log = get_logger("vidmcp.compositor")
ProgressFn = Callable[[float, str], None]


class CompositorEngine:
    def __init__(self, project: ProjectStore):
        self.project = project
        self.registry = get_effect_registry()

    def render(
        self,
        *,
        output_name: str | None = None,
        max_frames: int | None = None,
        progress: ProgressFn | None = None,
        preview_scale: int | None = None,
    ) -> dict:
        m = self.project.manifest
        if not m.source_video:
            raise RuntimeError("No source video in project")
        source = self.project.abs(m.source_video)
        meta = probe_video(source)
        layers = m.layers.sorted_layers()
        if not layers:
            raise RuntimeError("Layer stack empty — apply effects or add layers first")

        # prepare effects
        for L in layers:
            if L.effect:
                try:
                    eff = self.registry.get(L.effect.effect_type)
                    eff.prepare(L.effect, {"project": str(self.project.root)})
                except KeyError:
                    log.warning("unknown_effect_skipped_prepare", effect=L.effect.effect_type)

        render_id = str(uuid4())
        out_name = output_name or f"render_{render_id[:8]}.mp4"
        tmp_video = self.project.tmp_dir / f"{render_id}.mp4"
        final_path = self.project.renders_dir / out_name
        self.project.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.project.renders_dir.mkdir(parents=True, exist_ok=True)

        w, h = meta.width, meta.height
        if preview_scale and max(w, h) > preview_scale:
            scale = preview_scale / max(w, h)
            w, h = int(w * scale), int(h * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, meta.fps, (w, h))
        total = meta.frame_count if max_frames is None else min(meta.frame_count, max_frames)

        # resolve primary mask dir
        seg = m.primary_segment()
        mask_dir = str(self.project.abs(seg.mask_dir)) if seg else None

        for idx, frame in iter_frames(source):
            if max_frames is not None and idx >= max_frames:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            subject_mask = load_mask_for_frame(mask_dir, idx, h, w)
            if subject_mask is None:
                subject_mask = np.zeros((h, w), dtype=np.uint8)

            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            ts = idx / max(meta.fps, 1e-6)
            ctx = EffectContext(
                frame_index=idx,
                timestamp=ts,
                fps=meta.fps,
                width=w,
                height=h,
                subject_mask=subject_mask,
                source_frame=frame,
                project_dir=self.project.root,
            )

            for L in layers:
                canvas = self._composite_layer(canvas, L, ctx, frame, subject_mask)

            writer.write(canvas)
            if progress and total and idx % max(1, total // 25) == 0:
                progress(idx / total, f"composite {idx}/{total}")

        writer.release()

        # mux audio + h264
        h264_tmp = self.project.tmp_dir / f"{render_id}_h264.mp4"
        try:
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tmp_video),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    str(h264_tmp),
                ],
                check=True,
                capture_output=True,
            )
            mux_audio(h264_tmp, source, final_path)
        except Exception as e:  # noqa: BLE001
            log.warning("encode_mux_fallback", error=str(e))
            final_path.write_bytes(tmp_video.read_bytes())

        preview_path = self.project.previews_dir / f"preview_{render_id[:8]}.jpg"
        # last frame preview — re-open final
        cap = cv2.VideoCapture(str(final_path if final_path.exists() else tmp_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, (total // 2) if total else 0))
        ok, mid = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(preview_path), mid)

        rel_out = self.project.rel(final_path)
        rel_prev = self.project.rel(preview_path) if preview_path.exists() else None
        render_info = {
            "render_id": render_id,
            "output_path": rel_out,
            "preview_path": rel_prev,
            "width": w,
            "height": h,
            "fps": meta.fps,
            "frame_count": total,
            "layers_version": m.layers.version,
        }
        m.renders.append(render_info)
        m.status = ProjectStatus.RENDERED
        m.append_history("composite_and_render", render_info)
        self.project.save()
        if progress:
            progress(1.0, "render complete")
        return {
            **render_info,
            "absolute_output": str(final_path),
            "absolute_preview": str(preview_path) if preview_path.exists() else None,
            "duration_sec": total / max(meta.fps, 1e-6),
        }

    def _composite_layer(
        self,
        canvas: np.ndarray,
        layer: Layer,
        ctx: EffectContext,
        source_frame: np.ndarray,
        subject_mask: np.ndarray,
    ) -> np.ndarray:
        if layer.kind == LayerKind.SOURCE:
            return source_frame.copy() if canvas.max() == 0 else over(canvas, source_frame, np.full(subject_mask.shape, 255, np.uint8), layer.opacity)

        if layer.kind == LayerKind.BACKGROUND:
            bg = self._render_effect_or_asset(layer, ctx)
            # background only outside subject
            inv = 255 - to_u8_mask(subject_mask)
            if layer.mask_invert:
                inv = to_u8_mask(subject_mask)
            return over(canvas if canvas.max() else bg, bg, inv, layer.opacity)

        if layer.kind == LayerKind.SUBJECT:
            # subject cutout from source
            return over(canvas, source_frame, subject_mask, layer.opacity)

        if layer.kind == LayerKind.PARTICLES:
            part = self._render_effect_or_asset(layer, ctx)
            # particles only behind subject → outside mask
            inv = 255 - to_u8_mask(subject_mask)
            masked = over(np.zeros_like(part), part, inv, 1.0)
            if layer.blend_mode == BlendMode.SCREEN:
                return screen_blend(canvas, masked, layer.opacity)
            if layer.blend_mode == BlendMode.ADD:
                return add_blend(canvas, masked, layer.opacity)
            return over(canvas, masked, inv, layer.opacity)

        if layer.kind == LayerKind.BROLL:
            # full-frame under subject: treated like background plate from asset
            plate = self._load_layer_frame(layer, ctx)
            inv = 255 - to_u8_mask(subject_mask)
            return over(canvas if canvas.max() else plate, plate, inv, layer.opacity)

        if layer.kind in (LayerKind.GRADE, LayerKind.ADJUSTMENT, LayerKind.OVERLAY):
            graded = self._render_effect_or_asset(layer, ctx)
            # optional: grade only background
            selective = bool(layer.effect and layer.effect.params.get("background_only", False))
            if selective:
                inv = 255 - to_u8_mask(subject_mask)
                return over(canvas, graded, inv, layer.opacity)
            return cv2.addWeighted(canvas, 1.0 - layer.opacity, graded, layer.opacity, 0)

        return canvas

    def _render_effect_or_asset(self, layer: Layer, ctx: EffectContext) -> np.ndarray:
        if layer.effect:
            eff = self.registry.get(layer.effect.effect_type)
            return eff.render_frame(ctx, layer.effect)
        return self._load_layer_frame(layer, ctx)

    def _load_layer_frame(self, layer: Layer, ctx: EffectContext) -> np.ndarray:
        if not layer.asset_path:
            return np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)
        path = self.project.abs(layer.asset_path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)
            return cv2.resize(img, (ctx.width, ctx.height))
        # video plate
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, ctx.frame_index)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)
        return cv2.resize(frame, (ctx.width, ctx.height))
