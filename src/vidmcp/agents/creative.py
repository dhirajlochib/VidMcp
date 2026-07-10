"""Creative intent model — pacing curves and tone profiles made measurable.

Agents reason about 'feel' through numbers: target energy curve vs measured curve,
tone → concrete parameter mapping consumed by recipes and the plan drafter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.creative")

# target normalized energy at position x∈[0,1]
PACING_TEMPLATES: dict[str, Any] = {
    "hook_heavy_short": lambda x: 0.9 - 0.35 * x + 0.25 * (x > 0.85),
    "steady_educational": lambda x: 0.55 + 0.1 * np.sin(x * 6.28),
    "build_to_peak": lambda x: 0.35 + 0.6 * x**1.6,
    "wave": lambda x: 0.5 + 0.3 * np.sin(x * 9.42),
}

TONE_PROFILES: dict[str, dict[str, Any]] = {
    "energetic": {"lut": "vibrant_punch", "bgm_style": "uplifting", "sfx_density": "high",
                  "punch_in_gap_sec": 5.0, "caption_style": "karaoke", "cut_aggressiveness": 0.7},
    "calm": {"lut": "filmic_soft", "bgm_style": "lofi", "sfx_density": "low",
             "punch_in_gap_sec": 14.0, "caption_style": "minimal", "cut_aggressiveness": 0.35},
    "dramatic": {"lut": "teal_orange", "bgm_style": "tense", "sfx_density": "medium",
                 "punch_in_gap_sec": 8.0, "caption_style": "brand", "cut_aggressiveness": 0.5,
                 "keep_dramatic_pauses": True},
    "playful": {"lut": "vibrant_punch", "bgm_style": "playful", "sfx_density": "high",
                "punch_in_gap_sec": 6.0, "caption_style": "karaoke", "cut_aggressiveness": 0.65},
    "premium": {"lut": "cinematic_fade", "bgm_style": "cinematic", "sfx_density": "low",
                "punch_in_gap_sec": 12.0, "caption_style": "brand", "cut_aggressiveness": 0.45},
}


def set_creative_profile_project(
    project: Any,
    tone: str = "premium",
    pacing_template: str = "steady_educational",
) -> dict[str, Any]:
    if tone not in TONE_PROFILES:
        return {"ok": False, "message": f"Unknown tone '{tone}'. Available: {sorted(TONE_PROFILES)}"}
    if pacing_template not in PACING_TEMPLATES:
        return {"ok": False, "message": f"Unknown pacing '{pacing_template}'. Available: {sorted(PACING_TEMPLATES)}"}
    profile = {"tone": tone, "pacing_template": pacing_template, "params": TONE_PROFILES[tone]}
    project.manifest.analysis["creative_profile"] = profile
    project.manifest.append_history("set_creative_profile", {"tone": tone, "pacing": pacing_template})
    project.save()
    return {"ok": True, **profile}


def _measured_curve(project: Any, n_bins: int = 40) -> tuple[np.ndarray, float]:
    """Fuse audio energy + cut density + motion into a normalized pacing curve."""
    from vidmcp.perception.indexer import load_index

    index = load_index(project) or {}
    m = project.manifest
    duration = float(index.get("duration_sec") or (m.source_meta or {}).get("duration") or 60.0)
    curve = np.zeros(n_bins, np.float32)

    energy = index.get("energy") or []
    if energy:
        e = np.asarray(energy, np.float32)
        idx = np.linspace(0, len(e) - 1, n_bins).astype(int)
        curve += 0.6 * e[idx]

    shots = m.analysis.get("shots") or []
    if len(shots) > 1:
        cut_density = np.zeros(n_bins, np.float32)
        for s in shots:
            b = min(int(s["start"] / duration * n_bins), n_bins - 1)
            cut_density[b] += 1
        if cut_density.max() > 0:
            curve += 0.4 * cut_density / cut_density.max()
    else:
        curve += 0.2  # single shot baseline

    if curve.max() > 0:
        curve = curve / curve.max()
    return curve, duration


def analyze_pacing_project(project: Any) -> dict[str, Any]:
    measured, duration = _measured_curve(project)
    n = len(measured)
    profile = (project.manifest.analysis.get("creative_profile") or {})
    template = profile.get("pacing_template", "steady_educational")
    target_fn = PACING_TEMPLATES[template]
    xs = np.linspace(0, 1, n)
    target = np.array([float(target_fn(x)) for x in xs], np.float32)

    deviation = measured - target
    sags: list[dict[str, Any]] = []
    in_sag = None
    for i, d in enumerate(deviation):
        if d < -0.25 and in_sag is None:
            in_sag = i
        elif d >= -0.25 and in_sag is not None:
            t0, t1 = in_sag / n * duration, i / n * duration
            if t1 - t0 >= 8.0:
                sags.append({"t_start": round(t0, 1), "t_end": round(t1, 1),
                             "fix": "tighten cuts, add punch-in or B-roll here"})
            in_sag = None
    if in_sag is not None:
        t0 = in_sag / n * duration
        if duration - t0 >= 8.0:
            sags.append({"t_start": round(t0, 1), "t_end": round(duration, 1),
                         "fix": "ending sags — trim or add outro energy"})

    hook_bins = max(1, int(5.0 / duration * n))
    hook = float(measured[:hook_bins].mean())
    return {
        "ok": True,
        "pacing_template": template,
        "duration_sec": round(duration, 1),
        "hook_strength_first_5s": round(hook, 3),
        "mean_deviation": round(float(np.abs(deviation).mean()), 3),
        "sag_regions": sags,
        "curve_measured": [round(float(v), 2) for v in measured],
        "curve_target": [round(float(v), 2) for v in target],
        "verdict": "on-profile" if float(np.abs(deviation).mean()) < 0.22 and not sags else "needs-work",
    }
