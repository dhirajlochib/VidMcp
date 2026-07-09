"""Official Meta SAM 3 / SAM 3.1 video backend with Object Multiplex.

Prefers ``build_sam3_multiplex_video_predictor`` (SAM 3.1 shared-memory
multi-object tracking). Falls back to classic ``build_sam3_video_predictor``.

API surface follows facebookresearch/sam3:
  handle_request(start_session) → add_prompt(s) → propagate_in_video → close_session
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from vidmcp.perception.base import ObjectTrack, PerceptionBackend, ProgressFn, SegmentationResult
from vidmcp.perception.mask_ops import (
    coverage_mean,
    feather_mask,
    temporal_stability_score,
    to_u8_mask,
    union_masks,
)
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, write_mask_sequence

log = get_logger("vidmcp.perception.official")


def _import_builder():
    """Return (builder_fn, mode_name) preferring multiplex SAM 3.1."""
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        return build_sam3_multiplex_video_predictor, "sam3.1_multiplex"
    except ImportError:
        pass
    try:
        from sam3.model_builder import build_sam3_predictor

        def _build(**kw: Any):
            # version flag used by newer builders
            try:
                return build_sam3_predictor(version="sam3.1", **kw)
            except TypeError:
                return build_sam3_predictor(**kw)

        return _build, "sam3.1_predictor"
    except ImportError:
        pass
    from sam3.model_builder import build_sam3_video_predictor

    return build_sam3_video_predictor, "sam3_video"


def _tensor_to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class OfficialSam3Backend(PerceptionBackend):
    """Production SAM 3.1 Object Multiplex backend."""

    name = "official_sam3.1"

    def __init__(
        self,
        device: str = "auto",
        weights: Path | str | None = None,
        *,
        use_multiplex: bool = True,
        use_fa3: bool = False,
    ):
        self.device = device
        self.weights = Path(weights) if weights else None
        self.use_multiplex = use_multiplex
        self.use_fa3 = use_fa3
        self._predictor = None
        self._mode = "unloaded"

    def is_available(self) -> bool:
        try:
            _import_builder()
            return True
        except ImportError:
            return False

    def _build_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        from vidmcp.perception.weights import resolve_sam_weights

        builder, mode = _import_builder()
        if not self.use_multiplex and mode == "sam3.1_multiplex":
            from sam3.model_builder import build_sam3_video_predictor

            builder, mode = build_sam3_video_predictor, "sam3_video"

        # Resolve checkpoint from explicit path, env, HF cache, project weights/
        resolved = resolve_sam_weights(self.weights)
        if resolved is not None:
            self.weights = resolved
            log.info("sam_weights_resolved", path=str(resolved), size_mb=round(resolved.stat().st_size / 1e6, 1))
        elif self.weights:
            log.warning("sam_weights_missing", requested=str(self.weights))

        kwargs: dict[str, Any] = {}
        if self.weights and Path(self.weights).exists():
            # common kw names across builder versions — try all via progressive retry below
            kwargs["checkpoint_path"] = str(self.weights)
            kwargs["ckpt_path"] = str(self.weights)
            kwargs["checkpoint"] = str(self.weights)
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        kwargs["use_fa3"] = self.use_fa3

        # filter unsupported kwargs by progressive retry
        pred = None
        last_err: Exception | None = None
        for attempt_kwargs in (
            kwargs,
            {k: v for k, v in kwargs.items() if k in ("checkpoint_path", "device")},
            {k: v for k, v in kwargs.items() if k == "checkpoint_path"},
            {},
        ):
            try:
                pred = builder(**attempt_kwargs)
                break
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if pred is None:
            raise RuntimeError(f"Failed to build SAM predictor ({mode}): {last_err}")
        self._predictor = pred
        self._mode = mode
        log.info("sam_predictor_ready", mode=mode, weights=str(self.weights) if self.weights else None)
        return pred

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        feather: int = 3,
        **kwargs: Any,
    ) -> SegmentationResult:
        prompts = kwargs.get("prompts") or [prompt]
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.segment_multi(
            video_path,
            list(prompts),
            output_dir=output_dir,
            conf=conf,
            progress=progress,
            feather=feather,
            primary_prompt=prompt,
            frame_index=int(kwargs.get("frame_index", 0)),
            keep_per_object=bool(kwargs.get("keep_per_object", True)),
        )

    def segment_multi(
        self,
        video_path: Path,
        prompts: list[str],
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        feather: int = 3,
        primary_prompt: str | None = None,
        frame_index: int = 0,
        keep_per_object: bool = True,
    ) -> SegmentationResult:
        """Segment + track one or many text concepts using Object Multiplex when available."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = probe_video(video_path)
        primary_prompt = primary_prompt or (prompts[0] if prompts else "object")
        prompts = [p.strip() for p in prompts if p and str(p).strip()]
        if not prompts:
            prompts = [primary_prompt]

        if progress:
            progress(0.02, f"loading {self._mode or 'SAM 3.1'} predictor")
        predictor = self._build_predictor()

        session_id = None
        try:
            if progress:
                progress(0.05, "start_session")
            resp = self._request(
                predictor,
                type="start_session",
                resource_path=str(video_path),
            )
            session_id = resp.get("session_id") or resp.get("session")
            if not session_id:
                raise RuntimeError(f"start_session missing session_id: {list(resp.keys())}")

            # Optional reset for clean multi-class runs
            try:
                self._request(predictor, type="reset_session", session_id=session_id)
            except Exception:  # noqa: BLE001
                pass

            for i, text in enumerate(prompts):
                if progress:
                    progress(0.08 + 0.05 * (i / max(len(prompts), 1)), f"add_prompt: {text}")
                self._request(
                    predictor,
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=text,
                )

            if progress:
                progress(0.2, "propagate_in_video (Object Multiplex)")
            per_frame = list(self._propagate(predictor, session_id, meta.frame_count, progress))
            if not per_frame:
                # degraded: use last add_prompt outputs if any
                log.warning("propagate_empty_falling_back_static")
                per_frame = self._static_from_last(meta)

            # per-object dirs + union
            obj_masks: dict[int, list[np.ndarray]] = {}
            union_list: list[np.ndarray] = []
            label_map: dict[int, str] = {}

            for frame_idx, objs in per_frame:
                h, w = meta.height, meta.width
                frame_union = np.zeros((h, w), dtype=np.uint8)
                for obj_id, mask, score, label in objs:
                    m = feather_mask(mask, feather) if feather else to_u8_mask(mask)
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    frame_union = np.maximum(frame_union, m)
                    obj_masks.setdefault(obj_id, [])
                    # pad if frames skipped
                    while len(obj_masks[obj_id]) < frame_idx:
                        obj_masks[obj_id].append(np.zeros((h, w), dtype=np.uint8))
                    if len(obj_masks[obj_id]) == frame_idx:
                        obj_masks[obj_id].append(m)
                    else:
                        # overwrite / extend
                        if len(obj_masks[obj_id]) > frame_idx:
                            obj_masks[obj_id][frame_idx] = m
                        else:
                            obj_masks[obj_id].append(m)
                    if label:
                        label_map[obj_id] = label
                    elif obj_id not in label_map:
                        # map to prompt by index heuristic
                        label_map[obj_id] = prompts[min(obj_id - 1, len(prompts) - 1)] if obj_id >= 1 else primary_prompt
                while len(union_list) < frame_idx:
                    union_list.append(np.zeros((meta.height, meta.width), dtype=np.uint8))
                if len(union_list) == frame_idx:
                    union_list.append(frame_union)
                else:
                    if len(union_list) > frame_idx:
                        union_list[frame_idx] = frame_union
                    else:
                        union_list.append(frame_union)

            # ensure length matches
            target_n = max(meta.frame_count, len(union_list), 1)
            while len(union_list) < target_n:
                union_list.append(
                    union_list[-1].copy() if union_list else np.zeros((meta.height, meta.width), dtype=np.uint8)
                )

            write_mask_sequence(union_list, output_dir, prefix="mask")
            if keep_per_object and obj_masks:
                for oid, seq in obj_masks.items():
                    while len(seq) < len(union_list):
                        seq.append(np.zeros((meta.height, meta.width), dtype=np.uint8))
                    odir = output_dir / f"obj_{oid:03d}"
                    write_mask_sequence(seq[: len(union_list)], odir, prefix="mask")

            objects: list[ObjectTrack] = []
            for oid, seq in sorted(obj_masks.items()) or [(1, union_list)]:
                cov = coverage_mean(seq)
                objects.append(
                    ObjectTrack(
                        object_id=int(oid),
                        label=label_map.get(oid, primary_prompt),
                        confidence_mean=float(conf),
                        frame_span=(0, max(len(seq) - 1, 0)),
                        area_ratio_mean=cov,
                    )
                )
            if not objects:
                objects = [
                    ObjectTrack(
                        object_id=1,
                        label=primary_prompt,
                        confidence_mean=conf,
                        frame_span=(0, max(len(union_list) - 1, 0)),
                        area_ratio_mean=coverage_mean(union_list),
                    )
                ]

            stab = temporal_stability_score(union_list)
            cov = coverage_mean(union_list)
            if progress:
                progress(1.0, f"{self._mode} segment done · objs={len(objects)}")
            return SegmentationResult(
                prompt=primary_prompt,
                backend=f"{self.name}:{self._mode}",
                mask_dir=output_dir,
                masks=union_list,
                objects=objects,
                fps=meta.fps,
                width=meta.width,
                height=meta.height,
                frame_count=len(union_list),
                temporal_stability=stab,
                coverage_mean=cov,
                meta={
                    "session_id": session_id,
                    "mode": self._mode,
                    "multiplex": "multiplex" in self._mode,
                    "prompts": prompts,
                    "object_dirs": [f"obj_{o.object_id:03d}" for o in objects],
                    "per_object": keep_per_object,
                },
            )
        finally:
            if session_id is not None and self._predictor is not None:
                try:
                    self._request(self._predictor, type="close_session", session_id=session_id)
                except Exception:  # noqa: BLE001
                    try:
                        self._request(self._predictor, type="reset_session", session_id=session_id)
                    except Exception:  # noqa: BLE001
                        pass

    # ----- request / propagate helpers -----

    def _request(self, predictor: Any, **request: Any) -> dict[str, Any]:
        if hasattr(predictor, "handle_request"):
            out = predictor.handle_request(request=request)
            return out if isinstance(out, dict) else {"outputs": out}
        # method-style fallbacks
        typ = request.get("type")
        if typ == "start_session" and hasattr(predictor, "start_session"):
            return predictor.start_session(request.get("resource_path"))
        raise RuntimeError("Predictor has no handle_request")

    def _propagate(
        self,
        predictor: Any,
        session_id: str,
        frame_count: int,
        progress: ProgressFn | None,
    ) -> Iterator[tuple[int, list[tuple[int, np.ndarray, float, str | None]]]]:
        """Yield (frame_idx, [(obj_id, mask, score, label), ...])."""

        # 1) free function pattern from community: propagate_in_video(predictor, session_id)
        try:
            from sam3.model.sam3_video_predictor import propagate_in_video as prop_fn  # type: ignore
        except Exception:
            try:
                from sam3 import propagate_in_video as prop_fn  # type: ignore
            except Exception:
                prop_fn = None

        if prop_fn is not None:
            try:
                outputs = prop_fn(predictor, session_id)
                yield from self._normalize_propagate_outputs(outputs, frame_count, progress)
                return
            except Exception as e:  # noqa: BLE001
                log.warning("prop_fn_failed", error=str(e))

        # 2) method on predictor
        if hasattr(predictor, "propagate_in_video"):
            try:
                # signature variants
                for call in (
                    lambda: predictor.propagate_in_video(session_id),
                    lambda: predictor.propagate_in_video(session_id=session_id),
                    lambda: predictor.handle_request(
                        request=dict(type="propagate_in_video", session_id=session_id)
                    ),
                ):
                    try:
                        outputs = call()
                        yield from self._normalize_propagate_outputs(outputs, frame_count, progress)
                        return
                    except TypeError:
                        continue
            except Exception as e:  # noqa: BLE001
                log.warning("propagate_method_failed", error=str(e))

        # 3) handle_request propagate
        try:
            outputs = self._request(
                predictor,
                type="propagate_in_video",
                session_id=session_id,
            )
            yield from self._normalize_propagate_outputs(outputs, frame_count, progress)
            return
        except Exception as e:  # noqa: BLE001
            log.warning("handle_request_propagate_failed", error=str(e))

    def _normalize_propagate_outputs(
        self,
        outputs: Any,
        frame_count: int,
        progress: ProgressFn | None,
    ) -> Iterator[tuple[int, list[tuple[int, np.ndarray, float, str | None]]]]:
        if outputs is None:
            return
        # dict of frame_idx -> output
        if isinstance(outputs, dict) and outputs and all(
            isinstance(k, (int, str)) and str(k).isdigit() for k in list(outputs.keys())[:3]
        ):
            items = sorted(((int(k), v) for k, v in outputs.items()), key=lambda x: x[0])
            for i, (fi, val) in enumerate(items):
                yield fi, self._parse_frame_objects(val)
                if progress and frame_count:
                    progress(0.2 + 0.75 * ((i + 1) / max(len(items), 1)), f"frame {fi}")
            return

        # list / generator of frames
        if isinstance(outputs, dict) and "outputs" in outputs:
            outputs = outputs["outputs"]

        if isinstance(outputs, dict) and "per_frame_results" in outputs:
            outputs = outputs["per_frame_results"]

        try:
            iterator = iter(outputs)
        except TypeError:
            yield 0, self._parse_frame_objects(outputs)
            return

        for i, item in enumerate(iterator):
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], (int, np.integer)):
                fi = int(item[0])
                payload = item[1] if len(item) == 2 else item
                # (frame_idx, obj_ids, masks) SAM2-style
                if len(item) >= 3 and not isinstance(item[1], dict):
                    obj_ids = item[1]
                    masks = item[2]
                    objs = []
                    ids = list(obj_ids) if hasattr(obj_ids, "__iter__") else [1]
                    mask_arr = _tensor_to_numpy(masks)
                    for j, oid in enumerate(ids):
                        m = mask_arr[j] if mask_arr.ndim >= 3 and j < len(mask_arr) else mask_arr
                        objs.append((int(oid), to_u8_mask(m), 1.0, None))
                    yield fi, objs
                else:
                    yield fi, self._parse_frame_objects(payload)
            else:
                yield i, self._parse_frame_objects(item)
            if progress and frame_count:
                progress(0.2 + 0.75 * min(1.0, (i + 1) / max(frame_count, 1)), f"propagate {i}")

    def _parse_frame_objects(
        self, val: Any
    ) -> list[tuple[int, np.ndarray, float, str | None]]:
        if val is None:
            return []
        if isinstance(val, dict):
            masks = val.get("masks") or val.get("pred_masks") or val.get("out_binary_masks")
            obj_ids = val.get("obj_ids") or val.get("object_ids") or val.get("out_obj_ids")
            scores = val.get("scores") or val.get("object_score_logits") or val.get("out_probs")
            labels = val.get("labels") or val.get("phrases")
            if masks is None and "outputs" in val:
                return self._parse_frame_objects(val["outputs"])
            mask_arr = _tensor_to_numpy(masks) if masks is not None else None
            if mask_arr is None or mask_arr.size == 0:
                return []
            if mask_arr.ndim == 2:
                mask_arr = mask_arr[None, ...]
            n = mask_arr.shape[0]
            ids = list(_tensor_to_numpy(obj_ids).flatten()) if obj_ids is not None else list(range(1, n + 1))
            sc = list(_tensor_to_numpy(scores).flatten()) if scores is not None else [1.0] * n
            labs = list(labels) if labels is not None else [None] * n
            out = []
            for j in range(n):
                oid = int(ids[j]) if j < len(ids) else j + 1
                score = float(sc[j]) if j < len(sc) else 1.0
                lab = labs[j] if j < len(labs) else None
                out.append((oid, to_u8_mask(mask_arr[j]), score, str(lab) if lab else None))
            return out
        arr = _tensor_to_numpy(val)
        if arr.ndim >= 2:
            if arr.ndim == 2:
                return [(1, to_u8_mask(arr), 1.0, None)]
            return [(j + 1, to_u8_mask(arr[j]), 1.0, None) for j in range(arr.shape[0])]
        return []

    def _static_from_last(self, meta: Any) -> list[tuple[int, list]]:
        blank = np.zeros((meta.height, meta.width), dtype=np.uint8)
        return [(i, [(1, blank, 0.0, None)]) for i in range(max(meta.frame_count, 1))]
