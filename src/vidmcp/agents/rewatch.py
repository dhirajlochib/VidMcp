"""Rewatch QC — actually inspect the render: visual defects, audio defects, repair routes.

depth='mechanical': numeric checks. depth='full': also emits a frame contact sheet
for host-LLM perceptual review (via MCP resources/sampling on the host side).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.audio.loudness import target_for
from vidmcp.compositor.ffmpeg_ops import measure_loudness
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, sample_frames

log = get_logger("vidmcp.rewatch")

# defect kind → repair tool route
REPAIR_ROUTES: dict[str, str] = {
    "black_frames": "composite_and_render (re-render; check layer stack / max_frames)",
    "frozen_frames": "composite_and_render (source decode issue — re-render)",
    "luma_flicker": "stabilize_matte then composite_and_render",
    "clipping_video": "auto_color or apply_lut with lower intensity",
    "audio_clipping": "mixdown_audio (limiter re-engages) or process_audio",
    "lufs_offset": "mixdown_audio(target=...) or export via export_multi (loudness pass)",
    "dead_air": "plan_cuts + apply_cut_plan (remove dead air)",
    "av_sync": "composite_and_render (audio mux drift — re-render from mezzanine)",
    "too_dark": "auto_color(exposure=True)",
}


def _last_render(project: Any) -> Path | None:
    m = project.manifest
    for r in reversed(m.renders or []):
        p = project.abs(r.get("output_path"))
        if p and p.exists():
            return p
    return None


def _visual_qc(video: Path) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    frames = sample_frames(video, max_frames=48, max_side=480)
    if not frames:
        return [{"kind": "unreadable", "severity": "error", "t": 0.0, "detail": "cannot decode render"}]
    prev = None
    frozen_run = 0
    means = []
    for _, ts, img in frames:
        mean = float(img.mean())
        means.append((ts, mean))
        if mean < 4.0:
            defects.append({"kind": "black_frames", "severity": "error", "t": round(ts, 2),
                            "detail": f"frame mean {mean:.1f}"})
        if prev is not None:
            diff = float(np.abs(img.astype(np.int16) - prev.astype(np.int16)).mean())
            if diff < 0.05:
                frozen_run += 1
                if frozen_run == 3:
                    defects.append({"kind": "frozen_frames", "severity": "warning", "t": round(ts, 2),
                                    "detail": "3+ identical sampled frames"})
            else:
                frozen_run = 0
        prev = img
    # global luma flicker: alternating brightness jumps
    vals = np.array([v for _, v in means])
    if len(vals) > 6:
        jumps = np.abs(np.diff(vals))
        if float(np.median(jumps)) > 9.0:
            defects.append({"kind": "luma_flicker", "severity": "warning", "t": 0.0,
                            "detail": f"median inter-frame luma jump {float(np.median(jumps)):.1f}"})
    if float(np.mean(vals)) < 35:
        defects.append({"kind": "too_dark", "severity": "warning", "t": 0.0,
                        "detail": f"mean luma {float(np.mean(vals)):.0f}"})
    # clip stats via scopes on mid frame
    mid = frames[len(frames) // 2][2]
    from vidmcp.color.scopes import frame_stats

    st = frame_stats(mid)
    if st["clip_high_pct"] > 4.0:
        defects.append({"kind": "clipping_video", "severity": "warning", "t": frames[len(frames) // 2][1],
                        "detail": f"{st['clip_high_pct']}% highlights clipped"})
    return defects


def _audio_qc(project: Any, video: Path, target: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    meta = probe_video(video)
    stats: dict[str, Any] = {}
    if not meta.has_audio:
        return [{"kind": "dead_air", "severity": "error", "t": 0.0, "detail": "render has no audio stream"}], stats
    loud = measure_loudness(video)
    spec = target_for(target)
    stats["lufs"] = loud.get("input_i")
    stats["true_peak"] = loud.get("input_tp")
    if loud.get("input_i") is not None and abs(float(loud["input_i"]) - spec["lufs"]) > 1.5:
        defects.append({"kind": "lufs_offset", "severity": "warning", "t": 0.0,
                        "detail": f"LUFS {loud['input_i']} vs target {spec['lufs']}"})
    if loud.get("input_tp") is not None and float(loud["input_tp"]) > spec["tp"] + 0.3:
        defects.append({"kind": "audio_clipping", "severity": "warning", "t": 0.0,
                        "detail": f"true peak {loud['input_tp']} dBTP over {spec['tp']}"})
    # dead air: silencedetect
    try:
        out = subprocess.run(
            ["ffmpeg", "-i", str(video), "-af", "silencedetect=noise=-38dB:d=2.5", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        for line in out.stderr.splitlines():
            if "silence_start" in line:
                t = float(line.rsplit(":", 1)[1].strip())
                if t > 0.5:  # ignore leading silence
                    defects.append({"kind": "dead_air", "severity": "warning", "t": round(t, 2),
                                    "detail": "silence > 2.5s"})
    except Exception:  # noqa: BLE001
        pass
    return defects, stats


def _contact_sheet(project: Any, video: Path, cols: int = 4, rows: int = 3) -> str | None:
    frames = sample_frames(video, max_frames=cols * rows, max_side=320)
    if not frames:
        return None
    h, w = frames[0][2].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, (_, ts, img) in enumerate(frames[: cols * rows]):
        r, c = divmod(i, cols)
        img = cv2.resize(img, (w, h))
        cv2.putText(img, f"{ts:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    p = project.previews_dir / "rewatch_contact_sheet.jpg"
    cv2.imwrite(str(p), sheet)
    return project.rel(p)


def rewatch_project(project: Any, depth: str = "mechanical", target: str = "youtube") -> dict[str, Any]:
    video = _last_render(project)
    if video is None:
        return {"ok": False, "message": "No render to review — composite_and_render first"}
    defects = _visual_qc(video)
    audio_defects, audio_stats = _audio_qc(project, video, target)
    defects += audio_defects
    for d in defects:
        d["repair"] = REPAIR_ROUTES.get(d["kind"], "review manually")
    n_err = sum(1 for d in defects if d["severity"] == "error")
    score = max(0.0, 1.0 - 0.25 * n_err - 0.08 * (len(defects) - n_err))
    result: dict[str, Any] = {
        "ok": n_err == 0,
        "render": project.rel(video),
        "score": round(score, 3),
        "n_defects": len(defects),
        "defects": defects[:15],
        "audio": audio_stats,
        "loudness_target": target,
    }
    if depth == "full":
        sheet = _contact_sheet(project, video)
        result["contact_sheet"] = sheet
        result["host_review_hint"] = (
            "Read the contact sheet image + defects, judge cut smoothness / matte believability / "
            "graphic legibility, then call the repair tools listed per defect."
        )
    m = project.manifest
    m.reviews.append({"kind": "rewatch", "score": score, "n_defects": len(defects)})
    m.append_history("rewatch_render", {"score": score, "n_defects": len(defects)})
    project.save()
    return result
