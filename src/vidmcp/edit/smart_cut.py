"""Smart cut: remove dead air, long fillers, hesitation gaps."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.smart_cut")

DEFAULT_FILLERS = ("um", "uh", "uhh", "erm", "like", "you know", "basically", "actually")


@dataclass
class KeepRange:
    start: float
    end: float


def plan_smart_cuts(
    words: list[dict[str, Any]],
    *,
    duration_sec: float,
    min_gap: float = 0.45,
    fillers: tuple[str, ...] = DEFAULT_FILLERS,
    aggressiveness: float = 0.5,
) -> list[KeepRange]:
    """
    Return ranges to KEEP.
    aggressiveness 0..1: higher removes more (shorter min_gap, more fillers).
    """
    aggressiveness = max(0.0, min(1.0, float(aggressiveness)))
    gap_th = min_gap * (1.2 - 0.6 * aggressiveness)
    if not words:
        return [KeepRange(0.0, duration_sec)]

    # normalize words
    wlist = []
    for w in words:
        tok = str(w.get("word") or "").strip().lower().strip(".,!?")
        wlist.append(
            {
                "word": tok,
                "raw": str(w.get("word") or "").strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
        )
    wlist.sort(key=lambda x: x["start"])

    remove: list[tuple[float, float]] = []
    # filler tokens with long duration
    for w in wlist:
        dur = w["end"] - w["start"]
        if w["word"] in fillers and dur >= (0.35 - 0.15 * aggressiveness):
            remove.append((w["start"], w["end"]))
        # stretched single word hesitation
        if dur >= (0.9 - 0.3 * aggressiveness) and w["word"] in set(fillers) | {"this", "so", "and"}:
            remove.append((w["start"] + 0.05, w["end"]))

    # gaps between words
    for i in range(1, len(wlist)):
        gap = wlist[i]["start"] - wlist[i - 1]["end"]
        if gap >= gap_th:
            # keep 80ms pad
            a = wlist[i - 1]["end"] + 0.05
            b = wlist[i]["start"] - 0.05
            if b > a:
                remove.append((a, b))

    if not remove:
        return [KeepRange(0.0, duration_sec)]

    # merge remove intervals
    remove.sort()
    merged: list[list[float]] = []
    for a, b in remove:
        if not merged or a > merged[-1][1] + 0.02:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    # invert to keep
    keeps: list[KeepRange] = []
    cur = 0.0
    for a, b in merged:
        if a > cur + 0.02:
            keeps.append(KeepRange(cur, a))
        cur = max(cur, b)
    if cur < duration_sec - 0.02:
        keeps.append(KeepRange(cur, duration_sec))
    if not keeps:
        keeps = [KeepRange(0.0, duration_sec)]
    return keeps


def apply_smart_cuts(
    video: Path | str,
    out_video: Path | str,
    ranges: list[KeepRange],
    *,
    audio: Path | str | None = None,
    out_audio: Path | str | None = None,
) -> dict[str, Any]:
    """Concat keep-ranges from video (+ optional separate audio)."""
    video = Path(video)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video)
    if not ranges:
        ranges = [KeepRange(0.0, meta.duration_sec)]

    # Build filter_complex trim concat
    # Simpler approach: use ffmpeg select filter or multi-trim concat demuxer
    work = out_video.parent / "_smart_cut"
    work.mkdir(parents=True, exist_ok=True)
    list_file = work / "concat.txt"
    parts: list[Path] = []
    for i, r in enumerate(ranges):
        part = work / f"part_{i:03d}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{r.start:.3f}",
                "-to",
                f"{r.end:.3f}",
                "-i",
                str(video),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(part),
            ],
            check=True,
            capture_output=True,
        )
        parts.append(part)
    list_file.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(out_video),
        ],
        check=True,
        capture_output=True,
    )

    removed = meta.duration_sec - sum(r.end - r.start for r in ranges)
    out_meta = probe_video(out_video)
    result: dict[str, Any] = {
        "ok": True,
        "path": str(out_video),
        "ranges": [{"start": r.start, "end": r.end} for r in ranges],
        "duration_in": meta.duration_sec,
        "duration_out": out_meta.duration_sec,
        "removed_sec": max(0.0, removed),
    }

    if audio and out_audio:
        # cut same ranges on audio
        audio = Path(audio)
        out_audio = Path(out_audio)
        aparts = []
        for i, r in enumerate(ranges):
            ap = work / f"ap_{i:03d}.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{r.start:.3f}",
                    "-to",
                    f"{r.end:.3f}",
                    "-i",
                    str(audio),
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(ap),
                ],
                check=True,
                capture_output=True,
            )
            aparts.append(ap)
        alist = work / "aconcat.txt"
        alist.write_text("".join(f"file '{p}'\n" for p in aparts))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(alist),
                "-ar",
                "48000",
                "-ac",
                "2",
                str(out_audio),
            ],
            check=True,
            capture_output=True,
        )
        result["audio_path"] = str(out_audio)
    return result
