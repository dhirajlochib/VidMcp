"""Motion-reactive particle systems for behind-subject VFX."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams
from vidmcp.perception.mask_ops import to_u8_mask


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: tuple[int, int, int]


class ParticleSystemEffect(Effect):
    name = "particles"
    kind = "particles"

    def __init__(self) -> None:
        self._particles: list[Particle] = []
        self._initialized = False
        self._rng = np.random.default_rng(42)

    def prepare(self, params: EffectParams, ctx_meta: dict) -> None:
        self._particles = []
        self._initialized = False
        seed = int(params.params.get("seed", 42))
        self._rng = np.random.default_rng(seed)

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        style = str(params.params.get("style", "neon_dust"))
        density = float(params.params.get("density", 0.35)) * params.intensity
        canvas = np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)

        # spawn from subject silhouette edge when mask available
        spawn_points: list[tuple[int, int]] = []
        if ctx.subject_mask is not None:
            m = to_u8_mask(ctx.subject_mask)
            edges = cv2.Canny(m, 40, 120)
            ys, xs = np.where(edges > 0)
            if len(xs) > 0:
                n_sample = min(len(xs), max(8, int(80 * density)))
                idx = self._rng.choice(len(xs), size=n_sample, replace=False)
                spawn_points = list(zip(xs[idx].tolist(), ys[idx].tolist()))
        if not spawn_points:
            # ambient rain from top
            n = max(10, int(40 * density))
            spawn_points = [
                (int(self._rng.integers(0, ctx.width)), int(self._rng.integers(0, max(1, ctx.height // 5))))
                for _ in range(n)
            ]

        # spawn new particles
        n_new = max(1, int(15 * density))
        for i in range(min(n_new, len(spawn_points))):
            x, y = spawn_points[i % len(spawn_points)]
            if style == "sparks":
                vx, vy = float(self._rng.normal(0, 2)), float(self._rng.normal(-3, 1.5))
                color = (0, 180, 255)
                life = float(self._rng.uniform(8, 20))
                size = float(self._rng.uniform(1.5, 3.5))
            elif style == "rain":
                vx, vy = float(self._rng.normal(0, 0.5)), float(self._rng.uniform(6, 12))
                color = (220, 200, 160)
                life = float(self._rng.uniform(15, 40))
                size = float(self._rng.uniform(1.0, 2.0))
            else:  # neon_dust
                vx, vy = float(self._rng.normal(0, 1.2)), float(self._rng.normal(-1.0, 1.0))
                color = (255, 80, 220) if self._rng.random() > 0.5 else (255, 200, 40)
                life = float(self._rng.uniform(20, 50))
                size = float(self._rng.uniform(1.0, 2.8))
            self._particles.append(
                Particle(x=float(x), y=float(y), vx=vx, vy=vy, life=life, max_life=life, size=size, color=color)
            )

        # integrate
        alive: list[Particle] = []
        for p in self._particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            if p.life <= 0 or p.x < 0 or p.y < 0 or p.x >= ctx.width or p.y >= ctx.height:
                continue
            alpha = p.life / p.max_life
            rad = max(1, int(p.size))
            col = tuple(int(c * alpha) for c in p.color)
            cv2.circle(canvas, (int(p.x), int(p.y)), rad, col, -1, lineType=cv2.LINE_AA)
            if style == "sparks":
                cv2.line(
                    canvas,
                    (int(p.x), int(p.y)),
                    (int(p.x - p.vx * 2), int(p.y - p.vy * 2)),
                    col,
                    1,
                    lineType=cv2.LINE_AA,
                )
            alive.append(p)
        # limit particle count
        self._particles = alive[-500:]
        # glow
        glow = cv2.GaussianBlur(canvas, (0, 0), 3)
        return cv2.addWeighted(canvas, 1.0, glow, 0.8, 0)
