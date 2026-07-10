"""LUT engine — .cube parse/apply (trilinear) + procedural builtin looks.

Builtin looks are generated (original, no copyright). LUT applies as a grade layer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams, Layer, LayerKind
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.lut")


def parse_cube(path: Path | str) -> np.ndarray:
    """Parse .cube → (N,N,N,3) float32 table indexed [b][g][r]."""
    size = 0
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.upper().startswith(("TITLE", "DOMAIN")):
            continue
        if s.upper().startswith("LUT_3D_SIZE"):
            size = int(s.split()[-1])
            continue
        if s.upper().startswith("LUT_1D_SIZE"):
            size = -int(s.split()[-1])
            continue
        parts = s.split()
        if len(parts) == 3:
            rows.append([float(x) for x in parts])
    data = np.asarray(rows, dtype=np.float32)
    if size > 0:
        if len(data) != size**3:
            raise ValueError(f"cube size mismatch: {len(data)} rows for size {size}")
        return data.reshape(size, size, size, 3)  # [b][g][r]
    if size < 0:
        n = -size
        # promote 1D → 3D by independent channel curves
        table = np.zeros((n, n, n, 3), dtype=np.float32)
        r = data[:, 0]
        g = data[:, 1]
        b = data[:, 2]
        bi, gi, ri = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        table[..., 0] = r[ri]
        table[..., 1] = g[gi]
        table[..., 2] = b[bi]
        return table
    raise ValueError("No LUT_3D_SIZE/LUT_1D_SIZE in cube file")


def apply_lut(img_bgr: np.ndarray, table: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """Trilinear 3D LUT apply. img uint8 BGR, table [b][g][r] RGB-valued."""
    n = table.shape[0]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = rgb * (n - 1)
    x0 = np.floor(x).astype(np.int32)
    x1 = np.minimum(x0 + 1, n - 1)
    f = x - x0
    r0, g0, b0 = x0[..., 0], x0[..., 1], x0[..., 2]
    r1, g1, b1 = x1[..., 0], x1[..., 1], x1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    def T(bi, gi, ri):
        return table[bi, gi, ri]

    c000 = T(b0, g0, r0); c100 = T(b0, g0, r1)  # noqa: E702
    c010 = T(b0, g1, r0); c110 = T(b0, g1, r1)  # noqa: E702
    c001 = T(b1, g0, r0); c101 = T(b1, g0, r1)  # noqa: E702
    c011 = T(b1, g1, r0); c111 = T(b1, g1, r1)  # noqa: E702
    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    out = c0 * (1 - fb) + c1 * fb
    out = np.clip(out, 0, 1)
    if intensity < 1.0:
        out = rgb * (1 - intensity) + out * intensity
    return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Builtin procedural looks (rgb float in [0,1] → rgb)
# ---------------------------------------------------------------------------


def _curve(x: np.ndarray, lift: float, gamma: float, gain: float) -> np.ndarray:
    return np.clip((x**(1.0 / max(gamma, 1e-3))) * gain + lift, 0, 1)


def _split_tone(rgb: np.ndarray, shadow_rgb, high_rgb, amount: float) -> np.ndarray:
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sh = (1 - luma)[..., None] ** 2
    hi = luma[..., None] ** 2
    tint = sh * np.asarray(shadow_rgb, np.float32) + hi * np.asarray(high_rgb, np.float32)
    return np.clip(rgb + amount * (tint - 0.5) * 0.5, 0, 1)


def _sat(rgb: np.ndarray, s: float) -> np.ndarray:
    luma = (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    return np.clip(luma + (rgb - luma) * s, 0, 1)


BUILTIN_LOOKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "teal_orange": lambda x: _sat(_split_tone(x, (0.30, 0.55, 0.65), (0.75, 0.58, 0.38), 0.9), 1.08),
    "filmic_soft": lambda x: _curve(_sat(x, 0.94), 0.02, 1.06, 0.95),
    "noir": lambda x: _curve(_sat(x, 0.25), -0.02, 0.92, 1.1),
    "bleach_bypass": lambda x: _curve(_sat(x, 0.45), 0.0, 0.9, 1.15),
    "warm_portrait": lambda x: _sat(_split_tone(x, (0.5, 0.45, 0.42), (0.72, 0.62, 0.5), 0.7), 1.05),
    "cool_matte": lambda x: _curve(_split_tone(x, (0.42, 0.48, 0.58), (0.55, 0.6, 0.68), 0.6), 0.05, 1.02, 0.92),
    "vibrant_punch": lambda x: _curve(_sat(x, 1.28), 0.0, 1.05, 1.05),
    "cinematic_fade": lambda x: _curve(_sat(x, 0.88), 0.06, 1.1, 0.9),
}


def builtin_table(name: str, size: int = 33) -> np.ndarray:
    fn = BUILTIN_LOOKS.get(name)
    if fn is None:
        raise KeyError(f"Unknown builtin look '{name}'. Available: {sorted(BUILTIN_LOOKS)}")
    axis = np.linspace(0, 1, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)  # rgb values at [b][g][r]
    return fn(grid.reshape(-1, 3)).reshape(size, size, size, 3).astype(np.float32)


def write_cube(table: np.ndarray, path: Path | str, title: str = "vidmcp") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = table.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {n}"]
    flat = table.reshape(-1, 3)
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in flat]
    path.write_text("\n".join(lines))
    return path


_TABLE_CACHE: dict[str, np.ndarray] = {}


def resolve_table(lut: str) -> np.ndarray:
    if lut in _TABLE_CACHE:
        return _TABLE_CACHE[lut]
    if lut in BUILTIN_LOOKS:
        table = builtin_table(lut)
    else:
        p = Path(lut).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"LUT not found: {lut}. Builtins: {sorted(BUILTIN_LOOKS)}")
        table = parse_cube(p)
    _TABLE_CACHE[lut] = table
    return table


class LutEffect(Effect):
    name = "lut"
    kind = "grade"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        lut = str(params.params.get("lut") or params.params.get("name") or "filmic_soft")
        table = resolve_table(lut)
        return apply_lut(ctx.source_frame, table, float(params.intensity))


def list_luts(workspace_root: Path | None = None) -> dict[str, Any]:
    customs: list[str] = []
    if workspace_root:
        lut_dir = Path(workspace_root) / "luts"
        if lut_dir.exists():
            customs = [str(p) for p in sorted(lut_dir.glob("*.cube"))]
    return {"ok": True, "builtin": sorted(BUILTIN_LOOKS), "custom": customs}


def apply_lut_project(
    project: Any,
    lut: str = "filmic_soft",
    intensity: float = 1.0,
    background_only: bool = False,
) -> dict[str, Any]:
    resolve_table(lut)  # validate early
    m = project.manifest
    layer = m.layers.add(
        Layer(
            name=f"lut_{Path(lut).stem}",
            kind=LayerKind.GRADE,
            z_index=40,
            effect=EffectParams(
                effect_type="lut",
                intensity=float(np.clip(intensity, 0.0, 1.0)),
                params={"lut": lut, "background_only": background_only},
            ),
        )
    )
    m.append_history("apply_lut", {"lut": lut, "intensity": intensity})
    project.save()
    return {"ok": True, "layer_id": layer.id, "lut": lut, "intensity": intensity}
