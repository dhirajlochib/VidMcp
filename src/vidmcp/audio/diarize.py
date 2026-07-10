"""Multi-speaker diarization.

Uses energy + spectral clustering on MFCCs (sklearn if available) as offline path.
Optional pyannote when installed + HF token.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.diarize")


def _extract_wav(video: Path, wav: Path, sr: int = 16000) -> bool:
    try:
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", str(sr), "-vn", str(wav)],
            check=True,
            capture_output=True,
        )
        return wav.exists() and wav.stat().st_size > 44
    except Exception:
        return False


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio, sr


def _frame_features(audio: np.ndarray, sr: int, hop_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(sr * hop_ms / 1000))
    win = hop * 2
    feats = []
    times = []
    for i in range(0, max(len(audio) - win, 1), hop):
        seg = audio[i : i + win]
        if len(seg) < win:
            break
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
        # simple spectral centroid via FFT
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        centroid = float((spec * freqs).sum() / (spec.sum() + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)
        feats.append([rms, centroid / 8000.0, zcr])
        times.append(i / sr)
    if not feats:
        return np.zeros((0, 3)), np.zeros((0,))
    return np.array(feats, dtype=np.float32), np.array(times, dtype=np.float32)


def diarize_video(
    video_path: Path,
    *,
    work_dir: Path,
    n_speakers: int = 2,
    words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    wav = work_dir / "diarize.wav"
    if not _extract_wav(video_path, wav):
        # no audio — single speaker narrative
        segs = []
        if words:
            segs = [{"speaker": "SPEAKER_00", "start": words[0]["start"], "end": words[-1]["end"]}]
        return {
            "ok": True,
            "backend": "no_audio_single_speaker",
            "speakers": ["SPEAKER_00"],
            "segments": segs,
            "n_speakers": 1,
        }

    # try pyannote
    try:
        import os

        from pyannote.audio import Pipeline

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        diar = pipeline(str(wav))
        segments = []
        speakers = set()
        for turn, _, speaker in diar.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({"speaker": speaker, "start": float(turn.start), "end": float(turn.end)})
        return {
            "ok": True,
            "backend": "pyannote",
            "speakers": sorted(speakers),
            "segments": segments,
            "n_speakers": len(speakers),
        }
    except Exception as e:  # noqa: BLE001
        log.info("pyannote_unavailable", error=str(e)[:160])

    audio, sr = _load_wav(wav)
    feats, times = _frame_features(audio, sr)
    if len(feats) < 4:
        return {
            "ok": True,
            "backend": "insufficient_audio",
            "speakers": ["SPEAKER_00"],
            "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": float(len(audio) / sr)}],
            "n_speakers": 1,
        }

    # speech frames only
    speech_mask = feats[:, 0] > max(0.02, float(np.percentile(feats[:, 0], 40)))
    X = feats[speech_mask]
    t_speech = times[speech_mask]
    if len(X) < 4:
        X, t_speech = feats, times

    labels = _cluster(X, n_speakers=n_speakers)
    # merge contiguous
    segments = []
    if len(labels):
        cur = int(labels[0])
        start = float(t_speech[0])
        for i in range(1, len(labels)):
            if int(labels[i]) != cur or (t_speech[i] - t_speech[i - 1]) > 0.4:
                segments.append(
                    {
                        "speaker": f"SPEAKER_{cur:02d}",
                        "start": start,
                        "end": float(t_speech[i - 1] + 0.05),
                    }
                )
                cur = int(labels[i])
                start = float(t_speech[i])
        segments.append({"speaker": f"SPEAKER_{cur:02d}", "start": start, "end": float(t_speech[-1] + 0.05)})

    # assign words if provided
    word_speakers = []
    if words:
        for w in words:
            mid = 0.5 * (float(w["start"]) + float(w["end"]))
            sp = "SPEAKER_00"
            for seg in segments:
                if seg["start"] <= mid <= seg["end"]:
                    sp = seg["speaker"]
                    break
            word_speakers.append({**w, "speaker": sp})

    speakers = sorted({s["speaker"] for s in segments}) or ["SPEAKER_00"]
    return {
        "ok": True,
        "backend": "spectral_energy_cluster",
        "speakers": speakers,
        "segments": segments,
        "n_speakers": len(speakers),
        "words": word_speakers or None,
    }


def _cluster(X: np.ndarray, n_speakers: int = 2) -> np.ndarray:
    n_speakers = max(1, min(n_speakers, len(X)))
    try:
        from sklearn.cluster import KMeans

        # normalize
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
        km = KMeans(n_clusters=n_speakers, n_init=10, random_state=0)
        return km.fit_predict(Xn)
    except Exception:
        pass
    # fallback: threshold on spectral centroid
    c = X[:, 1]
    thr = float(np.median(c))
    return (c > thr).astype(int) if n_speakers > 1 else np.zeros(len(X), dtype=int)
