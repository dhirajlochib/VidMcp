"""Time warp — speed ramps with pitch-preserved speech, flow-interpolated slow-mo, freeze."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.timewarp")


def _atempo_chain(speed: float) -> str:
    """atempo only supports 0.5–2.0 per instance; chain for wider ranges."""
    parts = []
    s = float(speed)
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s *= 2.0
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def _speech_ranges(project: Any) -> list[tuple[float, float]]:
    try:
        from vidmcp.perception.indexer import load_index

        index = load_index(project) or {}
        return [(float(s["start"]), float(s["end"])) for s in index.get("sentences") or []]
    except Exception:  # noqa: BLE001
        return []


def _overlaps_speech(a: float, b: float, speech: list[tuple[float, float]]) -> bool:
    return any(not (b <= s or a >= e) for s, e in speech)


def time_warp_project(
    project: Any,
    ramps: list[dict[str, Any]],
    quality: str = "flow",
    allow_speech_warp: bool = False,
) -> dict[str, Any]:
    """ramps: [{t, speed}] keyframes — speed applies from t until the next keyframe.

    speed < 1 = slow motion (flow-interpolated when quality='flow'), > 1 = speedup.
    Speech regions are protected unless allow_speech_warp.
    """
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    src = project.abs(m.renders[-1]["output_path"]) if m.renders else project.abs(m.source_video)
    if not src.exists():
        src = project.abs(m.source_video)
    meta = probe_video(src)
    if not ramps:
        return {"ok": False, "message": "No ramps given: [{t, speed}, ...]"}

    ramps = sorted(({"t": float(r["t"]), "speed": float(r.get("speed", 1.0))} for r in ramps), key=lambda r: r["t"])
    if ramps[0]["t"] > 0:
        ramps.insert(0, {"t": 0.0, "speed": 1.0})
    speech = _speech_ranges(project) if not allow_speech_warp else []

    # build segments
    segments: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, r in enumerate(ramps):
        t0 = r["t"]
        t1 = ramps[i + 1]["t"] if i + 1 < len(ramps) else meta.duration_sec
        if t1 <= t0:
            continue
        speed = max(0.1, min(8.0, r["speed"]))
        if speed != 1.0 and _overlaps_speech(t0, t1, speech):
            warnings.append(f"segment {t0:.1f}-{t1:.1f}s overlaps speech — kept 1.0x (allow_speech_warp=True to override)")
            speed = 1.0
        segments.append({"t0": t0, "t1": t1, "speed": speed})

    work = project.tmp_dir / "timewarp"
    work.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for i, s in enumerate(segments):
        part = work / f"part_{i:03d}.mp4"
        vf = f"setpts={1.0 / s['speed']:.5f}*PTS"
        if s["speed"] < 0.75 and quality == "flow":
            target_fps = min(meta.fps * 2, 60)
            vf += f",minterpolate=fps={target_fps:.0f}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
        af = _atempo_chain(s["speed"])
        cmd = ["ffmpeg", "-y", "-ss", f"{s['t0']:.3f}", "-to", f"{s['t1']:.3f}", "-i", str(src),
               "-vf", vf, "-af", af,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
               str(part)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # minterpolate can fail on odd sizes — retry plain
            cmd[cmd.index("-vf") + 1] = f"setpts={1.0 / s['speed']:.5f}*PTS"
            subprocess.run(cmd, check=True, capture_output=True)
        parts.append(part)

    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    out = project.renders_dir / "timewarp.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", "-movflags", "+faststart", str(out)],
        check=True, capture_output=True,
    )
    shutil.rmtree(work, ignore_errors=True)
    out_meta = probe_video(out)
    rel = project.rel(out)
    m.renders.append({"render_id": "timewarp", "output_path": rel, "kind": "timewarp"})
    m.append_history("time_warp", {"segments": [(s["t0"], s["t1"], s["speed"]) for s in segments]})
    project.save()
    return {
        "ok": True,
        "output_path": rel,
        "duration_in": round(meta.duration_sec, 2),
        "duration_out": round(out_meta.duration_sec, 2),
        "segments": [{"t0": round(s["t0"], 2), "t1": round(s["t1"], 2), "speed": s["speed"]} for s in segments],
        "warnings": warnings,
        "quality": quality,
    }


def freeze_frame_project(project: Any, t: float, hold_sec: float = 1.5) -> dict[str, Any]:
    """Insert a freeze at t for hold_sec (audio keeps rolling? no — pads silence)."""
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    src = project.abs(m.renders[-1]["output_path"]) if m.renders else project.abs(m.source_video)
    out = project.renders_dir / f"freeze_{int(t)}s.mp4"
    vf = f"tpad=stop_mode=clone:stop_duration={hold_sec}"
    # split at t, freeze end of first part, concat — simpler: use select+tpad trick per segment
    work = project.tmp_dir / "freeze"
    work.mkdir(parents=True, exist_ok=True)
    a, b = work / "a.mp4", work / "b.mp4"
    subprocess.run(["ffmpeg", "-y", "-to", f"{t:.3f}", "-i", str(src), "-vf", vf,
                    "-af", f"apad=pad_dur={hold_sec}", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(a)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(b)], check=True, capture_output=True)
    listing = work / "c.txt"
    listing.write_text(f"file '{a}'\nfile '{b}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)],
                   check=True, capture_output=True)
    rel = project.rel(out)
    m.renders.append({"render_id": out.stem, "output_path": rel, "kind": "freeze"})
    project.save()
    return {"ok": True, "output_path": rel, "t": t, "hold_sec": hold_sec}
