"""Course compiler: intent → lesson script outline + scene beats + edit plan."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4


def compile_lesson(intent: str, *, duration_sec: float = 180.0, n_beats: int = 6) -> dict[str, Any]:
    """Heuristic course compiler (LLM host can expand); returns structured lesson graph."""
    topic = intent.strip() or "Mathematics"
    # extract topic keywords
    words = re.findall(r"[A-Za-z0-9]+", topic)
    title = topic[:80]
    beat_templates = [
        "Hook: pose the question",
        "Define key terms",
        "Build the core model / diagram",
        "Worked example step-by-step",
        "Common mistake / edge case",
        "Recap and takeaway",
        "Call to practice",
    ]
    n_beats = max(3, min(n_beats, len(beat_templates)))
    step_dur = duration_sec / n_beats
    beats = []
    for i in range(n_beats):
        beats.append(
            {
                "index": i,
                "t_start": round(i * step_dur, 2),
                "t_end": round((i + 1) * step_dur, 2),
                "title": beat_templates[i],
                "narration_cue": f"{beat_templates[i]} about {title}",
                "scene_prompt": f"{title}: {beat_templates[i]}",
                "manim_hint": "Write title; Create diagram; Indicate key formula",
                "fx": "particles" if i in (2, 3) else "none",
            }
        )
    lesson_id = str(uuid4())[:8]
    viddsl = _lesson_to_viddsl(title, beats)
    return {
        "ok": True,
        "lesson_id": lesson_id,
        "title": title,
        "duration_sec": duration_sec,
        "beats": beats,
        "keywords": words[:12],
        "viddsl": viddsl,
        "recommended_tools": [
            "segment_subject",
            "render_math_scene",
            "sync_audio_semantics",
            "refine_segment_keyframes",
            "composite_and_render",
            "run_critic_ensemble",
            "sign_render",
        ],
    }


def _lesson_to_viddsl(title: str, beats: list[dict]) -> str:
    lines = [
        f'// lesson: {title}',
        'track person as S',
        f'scene procedural("{title}") as B',
        'composite B under S',
    ]
    for b in beats:
        lines.append(f'// beat {b["index"]}: {b["title"]} @ {b["t_start"]}s')
    lines.append('gate stability >= 0.7')
    lines.append('sign')
    return "\n".join(lines) + "\n"
