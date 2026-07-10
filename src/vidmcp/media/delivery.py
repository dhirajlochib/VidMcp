"""Batch multi-platform delivery — one mezzanine, N targets with correct codec/loudness.

Applies the reframe crop track per aspect when available; falls back to scale+pad.
Hardware encode (VideoToolbox) used when present.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from vidmcp.audio.loudness import normalize_loudness, target_for
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.delivery")

TARGET_SPECS: dict[str, dict[str, Any]] = {
    "youtube_16x9": {"size": (1920, 1080), "aspect": "16:9", "crf": 18, "codec": "h264", "loudness": "youtube"},
    "youtube_4k": {"size": (3840, 2160), "aspect": "16:9", "crf": 17, "codec": "hevc", "loudness": "youtube"},
    "reels_9x16": {"size": (1080, 1920), "aspect": "9:16", "crf": 19, "codec": "h264", "loudness": "reels"},
    "tiktok_9x16": {"size": (1080, 1920), "aspect": "9:16", "crf": 19, "codec": "h264", "loudness": "tiktok"},
    "square_1x1": {"size": (1080, 1080), "aspect": "1:1", "crf": 19, "codec": "h264", "loudness": "reels"},
    "podcast_audio": {"size": None, "aspect": None, "codec": "audio", "loudness": "podcast"},
}


@lru_cache(maxsize=1)
def hw_encoder() -> str | None:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        if "h264_videotoolbox" in out.stdout:
            return "h264_videotoolbox"
    except Exception:  # noqa: BLE001
        pass
    return None


def _codec_args(spec: dict[str, Any], hw: bool) -> list[str]:
    if spec["codec"] == "audio":
        return ["-vn", "-c:a", "aac", "-b:a", "192k"]
    if hw and spec["codec"] == "h264" and hw_encoder():
        return ["-c:v", hw_encoder(), "-b:v", "10M", "-pix_fmt", "yuv420p"]
    codec = "libx265" if spec["codec"] == "hevc" else "libx264"
    return ["-c:v", codec, "-crf", str(spec.get("crf", 18)), "-pix_fmt", "yuv420p"]


def _mezzanine(project: Any) -> Path:
    m = project.manifest
    for r in reversed(m.renders or []):
        p = project.abs(r.get("output_path"))
        if p and p.exists():
            return p
    return project.abs(m.source_video)


def export_one(
    project: Any,
    mezz: Path,
    target: str,
    hw: bool = True,
) -> dict[str, Any]:
    spec = TARGET_SPECS.get(target)
    if spec is None:
        return {"ok": False, "target": target, "message": f"Unknown target. Available: {sorted(TARGET_SPECS)}"}
    out = project.renders_dir / f"delivery_{target}.{'m4a' if spec['codec'] == 'audio' else 'mp4'}"
    meta = probe_video(mezz)

    if spec["codec"] == "audio":
        subprocess.run(["ffmpeg", "-y", "-i", str(mezz), *(_codec_args(spec, hw)), str(out)],
                       check=True, capture_output=True)
    else:
        tw, th = spec["size"]
        src_aspect = meta.width / max(meta.height, 1)
        tgt_aspect = tw / th
        reframe_track = (project.manifest.analysis.get("reframe") or {}).get(spec["aspect"])
        if reframe_track and abs(src_aspect - tgt_aspect) > 0.05:
            # bake saliency crop path for this aspect
            from vidmcp.camera.reframe import _track_fn, render_crop_track

            render_crop_track(mezz, out, target_aspect=tgt_aspect,
                              center_fn=_track_fn(reframe_track), out_size=(tw, th))
        else:
            if abs(src_aspect - tgt_aspect) < 0.05:
                vf = f"scale={tw}:{th}"
            elif src_aspect > tgt_aspect:
                vf = f"crop=ih*{tgt_aspect:.6f}:ih,scale={tw}:{th}"  # center crop wide → tall
            else:
                vf = f"crop=iw:iw/{tgt_aspect:.6f},scale={tw}:{th}"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mezz), "-vf", vf, *(_codec_args(spec, hw)),
                 "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
                check=True, capture_output=True,
            )

    # loudness pass
    final = out.with_name(out.stem + "_ln" + out.suffix)
    try:
        norm = normalize_loudness(out, final, target=spec["loudness"], audio_only=spec["codec"] == "audio")
        out.unlink(missing_ok=True)
        final.rename(out)
        lufs, tp = norm.get("lufs_out"), norm.get("tp_out")
    except Exception as e:  # noqa: BLE001
        log.warning("loudness_pass_failed", target=target, error=str(e))
        lufs = tp = None
    size_mb = round(out.stat().st_size / 1e6, 1) if out.exists() else 0
    return {
        "ok": out.exists(),
        "target": target,
        "path": project.rel(out),
        "codec": spec["codec"],
        "size": spec.get("size"),
        "lufs": lufs,
        "true_peak": tp,
        "size_mb": size_mb,
        "loudness_spec": target_for(spec["loudness"]),
    }


def export_multi_project(
    project: Any,
    targets: list[str] | None = None,
    hw: bool = True,
) -> dict[str, Any]:
    targets = targets or ["youtube_16x9", "reels_9x16", "square_1x1"]
    mezz = _mezzanine(project)
    if not mezz.exists():
        return {"ok": False, "message": "No render or source video to export"}
    results = [export_one(project, mezz, t, hw=hw) for t in targets]
    ok = all(r.get("ok") for r in results)
    m = project.manifest
    for r in results:
        if r.get("ok"):
            m.renders.append({"render_id": f"delivery_{r['target']}", "output_path": r["path"], "kind": "delivery"})
    m.append_history("export_multi", {"targets": targets, "ok": ok})
    project.save()
    return {"ok": ok, "mezzanine": project.rel(mezz), "n_targets": len(results), "exports": results,
            "hw_encoder": hw_encoder() if hw else None}
