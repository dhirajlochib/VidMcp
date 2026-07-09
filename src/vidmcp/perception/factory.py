"""Backend selection: MLX SAM 3.1 (Apple Silicon) → official → ultralytics → mock."""

from __future__ import annotations

import platform
from pathlib import Path

from vidmcp.config import SamBackend, Settings, get_settings
from vidmcp.perception.base import PerceptionBackend
from vidmcp.perception.mock_backend import MockPerceptionBackend
from vidmcp.perception.mlx_backend import MLXSam31Backend
from vidmcp.perception.official_backend import OfficialSam3Backend
from vidmcp.perception.ultralytics_backend import UltralyticsSam3Backend
from vidmcp.perception.weights import resolve_sam_weights
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.perception.factory")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_perception_backend(settings: Settings | None = None) -> PerceptionBackend:
    settings = settings or get_settings()
    mode = settings.sam_backend
    weights = resolve_sam_weights(settings.sam_weights) or settings.sam_weights
    device = settings.device

    def _mlx() -> MLXSam31Backend:
        return MLXSam31Backend(
            model_id=getattr(settings, "mlx_model_id", "mlx-community/sam3.1-bf16"),
            score_threshold=settings.conf_threshold,
            detect_every=int(getattr(settings, "mlx_detect_every", 8) or 8),
            max_side=getattr(settings, "mlx_max_side", 768),
        )

    if mode == SamBackend.MOCK:
        log.info("backend_selected", backend="mock", reason="config")
        return MockPerceptionBackend()

    if mode == SamBackend.MLX:
        b = _mlx()
        if not b.is_available():
            raise RuntimeError("MLX backend requested but mlx-vlm is not installed. pip install 'mlx-vlm>=0.4.3'")
        return b

    if mode == SamBackend.OFFICIAL:
        b = OfficialSam3Backend(
            device=device,
            weights=weights,
            use_multiplex=settings.sam_use_multiplex,
            use_fa3=settings.sam_use_fa3,
        )
        if not b.is_available():
            raise RuntimeError("Official sam3 package not installed.")
        return b

    if mode == SamBackend.ULTRALYTICS:
        return UltralyticsSam3Backend(weights=weights, device=device)

    # AUTO preference order on Apple Silicon: MLX first
    if _is_apple_silicon():
        mlx = _mlx()
        if mlx.is_available():
            log.info(
                "backend_selected",
                backend="mlx_sam3.1",
                reason="auto_apple_silicon",
                model=getattr(settings, "mlx_model_id", None),
            )
            return mlx

    official = OfficialSam3Backend(
        device=device,
        weights=weights,
        use_multiplex=settings.sam_use_multiplex,
        use_fa3=settings.sam_use_fa3,
    )
    if official.is_available():
        log.info("backend_selected", backend="official_sam3.1", reason="auto")
        return official

    ultra = UltralyticsSam3Backend(weights=weights, device=device)
    if ultra.is_available():
        wpath = Path(weights) if weights else ultra.weights
        if wpath and Path(wpath).exists():
            log.info("backend_selected", backend="ultralytics", reason="auto+weights")
            return ultra
        log.warning("ultralytics_present_but_no_weights_falling_back")

    # MLX even if not Apple (won't work well but try)
    mlx = _mlx()
    if mlx.is_available() and _is_apple_silicon():
        return mlx

    log.info("backend_selected", backend="mock", reason="auto_fallback")
    return MockPerceptionBackend()
