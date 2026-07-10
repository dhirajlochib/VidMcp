"""Footage index — one analysis pass extracting everything agents search/plan against.

Visual stats + faces, audio event tags, speech words/sentences, per-second energy curve.
All dependency-free (cv2/numpy/ffmpeg); optional models upgrade quality transparently.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson

from vidmcp.audio.media import extract_wav, has_audio_stream
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, sample_frames

log = get_logger("vidmcp.indexer")

INDEX_REL = "index/footage_index.json"


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    from vidmcp.perception.faces import detect_faces

    return detect_faces(gray)


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
    return data, sr


def _audio_events(samples: np.ndarray, sr: int) -> list[dict[str, Any]]:
    """Per-second tags: silence | speech | music | burst (laughter/applause-like)."""
    win = sr
    events = []
    n_sec = len(samples) // win
    for i in range(n_sec):
        seg = samples[i * win : (i + 1) * win]
        rms = float(np.sqrt((seg**2).mean()) + 1e-9)
        # sub-window energies at 8 Hz for burstiness
        sub = seg[: (len(seg) // 8) * 8].reshape(8, -1)
        sub_rms = np.sqrt((sub**2).mean(axis=1))
        burstiness = float(sub_rms.std() / (sub_rms.mean() + 1e-9))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        centroid = float((spec * freqs).sum() / (spec.sum() + 1e-9))
        flatness = float(np.exp(np.log(spec + 1e-9).mean()) / (spec.mean() + 1e-9))
        if rms < 0.008:
            tag = "silence"
        elif burstiness > 0.85 and rms > 0.05:
            tag = "burst"  # laughter / applause / impact-like
        elif flatness < 0.12 and 400 < centroid < 3000 and burstiness > 0.3:
            tag = "speech"
        elif flatness < 0.25 and burstiness < 0.35:
            tag = "music"
        else:
            tag = "noise"
        events.append({"t": i, "tag": tag, "rms": round(rms, 5), "centroid": int(centroid)})
    return events


def _pitch_track(samples: np.ndarray, sr: int) -> list[float]:
    """Per-second fundamental estimate via autocorrelation (prosody proxy)."""
    win = sr
    out = []
    for i in range(len(samples) // win):
        seg = samples[i * win : i * win + min(win, 2048 * 8)]
        seg = seg - seg.mean()
        if np.abs(seg).max() < 0.01:
            out.append(0.0)
            continue
        corr = np.correlate(seg[:2048], seg[:2048], mode="full")[2048:]
        lo, hi = sr // 400, sr // 70  # 70–400 Hz
        if hi >= len(corr):
            out.append(0.0)
            continue
        peak = int(np.argmax(corr[lo:hi])) + lo
        out.append(float(sr / peak) if corr[peak] > 0.15 * corr[0] else 0.0)
    return out


def _sentences_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sents: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        cur.append(w)
        gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 99.0
        text = str(w.get("word") or "")
        if gap > 0.8 or text.rstrip().endswith((".", "!", "?")):
            sents.append(
                {
                    "text": " ".join(str(x.get("word") or "").strip() for x in cur),
                    "start": float(cur[0]["start"]),
                    "end": float(cur[-1]["end"]),
                }
            )
            cur = []
    if cur:
        sents.append(
            {
                "text": " ".join(str(x.get("word") or "").strip() for x in cur),
                "start": float(cur[0]["start"]),
                "end": float(cur[-1]["end"]),
            }
        )
    return sents


def build_index_project(
    project: Any,
    include: list[str] | None = None,
    model_size: str = "base",
    fallback_transcript: str | None = None,
) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    include = include or ["visual", "audio_events", "speech", "emotion"]
    video = project.abs(m.source_video)
    meta = probe_video(video)
    index: dict[str, Any] = {"duration_sec": meta.duration_sec, "fps": meta.fps}

    # --- visual ---
    if "visual" in include:
        frames = sample_frames(video, max_frames=min(240, max(24, int(meta.duration_sec / 2))), max_side=480)
        visual = []
        for idx, ts, img in frames:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            bright = float(gray.mean())
            faces = [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in _detect_faces(gray)]
            visual.append(
                {
                    "t": round(ts, 2),
                    "frame": idx,
                    "sharpness": round(sharp, 1),
                    "brightness": round(bright, 1),
                    "n_faces": len(faces),
                    "face_area": round(sum(f["w"] * f["h"] for f in faces) / (img.shape[0] * img.shape[1]), 4),
                    "faces": faces[:3],
                }
            )
        index["visual"] = visual

    # --- audio ---
    samples = None
    sr = 16000
    if has_audio_stream(video) and ("audio_events" in include or "emotion" in include):
        wav = project.tmp_dir / "index_audio.wav"
        try:
            extract_wav(video, wav, sr=sr)
            samples, sr = _load_wav_mono(wav)
        except Exception as e:  # noqa: BLE001
            log.warning("index_audio_failed", error=str(e))
    if samples is not None and "audio_events" in include:
        # optional PANNs upgrade
        events = _audio_events(samples, sr)
        index["audio_events"] = events
        index["event_tags"] = sorted({e["tag"] for e in events})

    # --- speech ---
    if "speech" in include:
        try:
            from vidmcp.audio.whisper_timeline import transcribe_words

            tw = transcribe_words(
                video,
                work_dir=project.tmp_dir / "index_asr",
                model_size=model_size,
                fallback_transcript=fallback_transcript,
                auto_narrate_if_silent=False,
            )
            words = tw.get("words") or []
            index["transcript"] = tw.get("text") or " ".join(str(w.get("word") or "") for w in words)
            index["words"] = words
            index["sentences"] = _sentences_from_words(words)
            index["asr_backend"] = tw.get("backend")
        except Exception as e:  # noqa: BLE001
            log.warning("index_speech_failed", error=str(e))
            index["sentences"] = []

    # --- emotion / energy curve ---
    if "emotion" in include and samples is not None:
        win = sr
        n_sec = len(samples) // win
        rms = np.array(
            [float(np.sqrt((samples[i * win : (i + 1) * win] ** 2).mean())) for i in range(n_sec)]
        )
        pitch = np.array(_pitch_track(samples, sr))
        e_norm = rms / (rms.max() + 1e-9)
        p_var = np.zeros_like(e_norm)
        for i in range(len(pitch)):
            lo, hi = max(0, i - 2), min(len(pitch), i + 3)
            window = pitch[lo:hi]
            window = window[window > 0]
            p_var[i] = window.std() / (window.mean() + 1e-9) if window.size > 1 else 0.0
        energy = np.clip(0.7 * e_norm + 0.3 * np.clip(p_var * 3, 0, 1), 0, 1)
        labels = []
        for e, pv in zip(energy, p_var):
            if e > 0.65:
                labels.append("excited")
            elif e < 0.15:
                labels.append("quiet")
            elif pv > 0.25:
                labels.append("animated")
            else:
                labels.append("calm")
        index["energy"] = [round(float(x), 3) for x in energy]
        index["emotion"] = labels
        peaks = [int(i) for i in np.argsort(energy)[::-1][:8]]
        index["energy_peaks_sec"] = sorted(peaks)

    out_path = project.root / INDEX_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(index, option=orjson.OPT_INDENT_2))
    stats = {
        "n_visual": len(index.get("visual") or []),
        "n_words": len(index.get("words") or []),
        "n_sentences": len(index.get("sentences") or []),
        "n_audio_events": len(index.get("audio_events") or []),
        "event_tags": index.get("event_tags"),
    }
    m.analysis["footage_index"] = {"path": INDEX_REL, **stats}
    m.append_history("build_footage_index", stats)
    project.save()
    return {"ok": True, "index_path": INDEX_REL, **stats}


def load_index(project: Any) -> dict[str, Any] | None:
    p = project.root / INDEX_REL
    if not p.exists():
        return None
    try:
        return orjson.loads(p.read_bytes())
    except Exception:  # noqa: BLE001
        return None
