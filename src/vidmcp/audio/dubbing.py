"""Dubbing pipeline — segment TTS with duration fitting → dub track → render variant.

Translation happens at the host LLM (pass translated_segments); this module handles
timing-fit synthesis and assembly. Voice cloning is consent-gated and logged.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from vidmcp.audio.tracks import SR, _load_stereo, _write_stereo, set_track
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.dubbing")

_SAY_VOICES = {
    "es": "Monica", "fr": "Thomas", "de": "Anna", "it": "Alice", "pt": "Luciana",
    "hi": "Lekha", "ja": "Kyoko", "ko": "Yuna", "zh": "Tingting", "en": "Samantha",
}


def _tts_say(text: str, out_wav: Path, language: str) -> bool:
    if platform.system() != "Darwin" or not shutil.which("say"):
        return False
    voice = _SAY_VOICES.get(language[:2].lower())
    aiff = out_wav.with_suffix(".aiff")
    cmd = ["say", "-o", str(aiff)]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff), "-ar", str(SR), "-ac", "2", str(out_wav)],
            check=True, capture_output=True,
        )
        aiff.unlink(missing_ok=True)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("say_tts_failed", error=str(e))
        return False


def _tts_espeak(text: str, out_wav: Path, language: str) -> bool:
    if not shutil.which("espeak") and not shutil.which("espeak-ng"):
        return False
    binname = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
    try:
        raw = out_wav.with_suffix(".raw.wav")
        subprocess.run(
            [binname, "-v", language[:2].lower(), "-w", str(raw), text],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw), "-ar", str(SR), "-ac", "2", str(out_wav)],
            check=True, capture_output=True,
        )
        raw.unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _fit_duration(wav: Path, target_sec: float, out: Path, max_stretch: float = 0.15) -> float:
    """Time-stretch (atempo, pitch-safe-ish for ≤±15%) to fit the original segment slot."""
    from vidmcp.utils.video_io import probe_video

    try:
        cur = probe_video(wav).duration_sec
    except Exception:  # noqa: BLE001
        cur = target_sec
    if cur <= 0 or target_sec <= 0:
        shutil.copy2(wav, out)
        return cur
    ratio = cur / target_sec
    ratio = float(np.clip(ratio, 1 - max_stretch, 1 + max_stretch)) if abs(ratio - 1) > max_stretch else ratio
    if abs(ratio - 1.0) < 0.02:
        shutil.copy2(wav, out)
        return cur
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-af", f"atempo={ratio:.4f}", "-ar", str(SR), str(out)],
        check=True, capture_output=True,
    )
    return cur / ratio


def dub_video_project(
    project: Any,
    language: str,
    translated_segments: list[dict[str, Any]] | None = None,
    voice: str = "neutral",
    voice_clone_consent: bool = False,
    render: bool = True,
) -> dict[str, Any]:
    m = project.manifest
    if voice == "clone" and not voice_clone_consent:
        return {
            "ok": False,
            "message": "Voice cloning requires voice_clone_consent=True (explicit speaker consent). "
                       "Use voice='neutral' for standard TTS.",
        }
    if voice == "clone":
        from vidmcp.models_registry import ensure_model

        xtts = ensure_model("xtts")
        if not xtts.get("found"):
            return {"ok": False, "message": f"XTTS not installed: {xtts.get('hint')}. Falling back is disabled for cloning."}
        return {"ok": False, "message": "XTTS cloning adapter not yet wired — use voice='neutral'"}

    if translated_segments is None:
        from vidmcp.perception.indexer import load_index

        index = load_index(project) or {}
        sents = index.get("sentences") or []
        if not sents:
            return {"ok": False, "message": "No sentences — run build_footage_index, or pass translated_segments "
                                            "[{start, end, text}] (host LLM translates)"}
        translated_segments = sents  # same-language re-voice
    duration = max(float(s["end"]) for s in translated_segments) + 1.0

    work = project.tmp_dir / f"dub_{language}"
    work.mkdir(parents=True, exist_ok=True)
    timeline = np.zeros((int(duration * SR), 2), np.float32)
    synthesized = 0
    for i, seg in enumerate(translated_segments):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        raw = work / f"seg_{i:03d}_raw.wav"
        fit = work / f"seg_{i:03d}.wav"
        ok = _tts_say(text, raw, language) or _tts_espeak(text, raw, language)
        if not ok:
            return {"ok": False, "message": "No TTS backend (macOS say / espeak). Install espeak-ng."}
        slot = float(seg["end"]) - float(seg["start"])
        _fit_duration(raw, slot, fit)
        buf = _load_stereo(fit)
        start = int(float(seg["start"]) * SR)
        end = min(len(timeline), start + len(buf))
        timeline[start:end] += buf[: end - start]
        synthesized += 1

    dub_wav = project.root / "audio" / f"dub_{language}.wav"
    _write_stereo(timeline, dub_wav)
    set_track(project, f"dub_{language}", project.rel(dub_wav), gain_db=0.0, meta={"language": language})
    m.append_history("dub_video", {"language": language, "segments": synthesized})
    project.save()

    out: dict[str, Any] = {
        "ok": True,
        "language": language,
        "n_segments": synthesized,
        "dub_wav": project.rel(dub_wav),
    }
    if render and m.source_video:
        video = project.abs(m.source_video)
        dubbed = project.renders_dir / f"dub_{language}.mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video), "-i", str(dub_wav),
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-shortest", str(dubbed)],
                check=True, capture_output=True,
            )
            rel = project.rel(dubbed)
            m.renders.append({"render_id": f"dub_{language}", "output_path": rel, "kind": "dub"})
            project.save()
            out["render"] = rel
        except Exception as e:  # noqa: BLE001
            out["render_error"] = str(e)
    return out
