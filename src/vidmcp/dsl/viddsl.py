"""VidDSL — tiny typed recipe language for auditable video edits.

Grammar (line-oriented):
  track <prompt> as <NAME>
  scene procedural("...") as <NAME>
  scene manim("...") as <NAME>
  composite <BG> under <SUBJECT>
  effect <type> [params]
  gate stability >= 0.7
  refine
  sign
  variants <n>
  // comment
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DSLOp:
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class DSLProgram:
    ops: list[DSLOp] = field(default_factory=list)
    symbols: dict[str, str] = field(default_factory=dict)  # name -> kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "ops": [{"op": o.op, "args": o.args, "raw": o.raw} for o in self.ops],
            "symbols": self.symbols,
        }


_TRACK = re.compile(r'^track\s+(.+?)\s+as\s+(\w+)\s*$', re.I)
_SCENE = re.compile(r'^scene\s+(procedural|manim)\s*\(\s*"(.*)"\s*\)\s+as\s+(\w+)\s*$', re.I)
_COMP = re.compile(r'^composite\s+(\w+)\s+under\s+(\w+)\s*$', re.I)
_EFFECT = re.compile(r'^effect\s+(\w+)(?:\s+(.*))?$', re.I)
_GATE = re.compile(r'^gate\s+(\w+)\s*(>=|<=|>|<|==)\s*([0-9.]+)\s*$', re.I)
_VARIANTS = re.compile(r'^variants\s+(\d+)\s*$', re.I)


def compile_viddsl(source: str) -> DSLProgram:
    prog = DSLProgram()
    for line_no, line in enumerate(source.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue
        if s.lower() in {"refine", "sign", "critic", "uncertainty"}:
            prog.ops.append(DSLOp(op=s.lower(), raw=s))
            continue
        m = _TRACK.match(s)
        if m:
            prompt, name = m.group(1).strip().strip('"'), m.group(2)
            prog.symbols[name] = "track"
            prog.ops.append(DSLOp("track", {"prompt": prompt, "name": name}, s))
            continue
        m = _SCENE.match(s)
        if m:
            engine, prompt, name = m.group(1).lower(), m.group(2), m.group(3)
            prog.symbols[name] = "scene"
            prog.ops.append(DSLOp("scene", {"engine": engine, "prompt": prompt, "name": name}, s))
            continue
        m = _COMP.match(s)
        if m:
            prog.ops.append(DSLOp("composite", {"bg": m.group(1), "subject": m.group(2)}, s))
            continue
        m = _EFFECT.match(s)
        if m:
            params = {}
            if m.group(2):
                for part in m.group(2).split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k] = float(v) if re.match(r"^[0-9.]+$", v) else v
            prog.ops.append(DSLOp("effect", {"type": m.group(1), "params": params}, s))
            continue
        m = _GATE.match(s)
        if m:
            prog.ops.append(
                DSLOp("gate", {"metric": m.group(1), "op": m.group(2), "value": float(m.group(3))}, s)
            )
            continue
        m = _VARIANTS.match(s)
        if m:
            prog.ops.append(DSLOp("variants", {"n": int(m.group(1))}, s))
            continue
        raise ValueError(f"VidDSL parse error line {line_no}: {s}")
    return prog


def run_viddsl(
    program: DSLProgram | str,
    *,
    project,
    service_module,
    max_render_frames: int | None = None,
) -> dict[str, Any]:
    """Execute compiled DSL against a ProjectStore using service layer."""
    if isinstance(program, str):
        program = compile_viddsl(program)
    results = []
    bindings: dict[str, Any] = {}
    for op in program.ops:
        r: dict[str, Any] = {"op": op.op}
        if op.op == "track":
            seg = service_module.segment(project, prompt=op.args["prompt"])
            bindings[op.args["name"]] = {"type": "track", "segment": seg}
            r["result"] = {"segment_id": seg.get("segment_id")}
        elif op.op == "scene":
            eng = "manim" if op.args["engine"] == "manim" else "procedural"
            sc = service_module.render_math_scene(
                project, prompt=op.args["prompt"], engine=eng, place_as_background=True
            )
            bindings[op.args["name"]] = {"type": "scene", "scene": sc}
            r["result"] = sc
        elif op.op == "effect":
            specs = [
                {
                    "effect_type": op.args["type"],
                    "kind": "particles" if op.args["type"] == "particles" else "background",
                    "params": op.args.get("params") or {},
                    "name": op.args["type"],
                }
            ]
            r["result"] = service_module.apply_effects(project, effect_specs=specs, replace_existing=False)
        elif op.op == "composite":
            r["result"] = service_module.composite(project, max_frames=max_render_frames)
        elif op.op == "refine":
            r["result"] = service_module.refine_segment_keyframes(project, auto_detect=True)
        elif op.op == "gate":
            from vidmcp.harness.quality_gates import evaluate_gates

            gate = evaluate_gates(project)
            metric = op.args["metric"]
            val = gate.score if metric in ("stability", "score") else gate.score
            # map stability to temporal if available
            cmp_op = op.args["op"]
            thr = op.args["value"]
            ok = {
                ">=": val >= thr,
                "<=": val <= thr,
                ">": val > thr,
                "<": val < thr,
                "==": abs(val - thr) < 1e-6,
            }[cmp_op]
            r["result"] = {"passed": ok, "value": val, "threshold": thr, "gate": gate.to_dict()}
            if not ok:
                r["failed"] = True
        elif op.op == "sign":
            from vidmcp.provenance.sign import sign_project_render

            r["result"] = sign_project_render(project)
        elif op.op == "critic":
            from vidmcp.critics.ensemble import run_critic_ensemble

            r["result"] = run_critic_ensemble(project)
        elif op.op == "uncertainty":
            from vidmcp.advanced.uncertainty import compute_uncertainty_field

            seg = project.manifest.primary_segment()
            if not seg:
                r["result"] = {"ok": False, "message": "no segment"}
            else:
                r["result"] = compute_uncertainty_field(
                    project.abs(seg.mask_dir), out_dir=project.previews_dir / "uncertainty"
                )
        elif op.op == "variants":
            from vidmcp.config import get_settings
            from vidmcp.core.workspace import Workspace
            from vidmcp.harness.runtime import HarnessRuntime

            rt = HarnessRuntime(Workspace(get_settings()), get_settings())
            r["result"] = rt.generate_variants(
                project.manifest.id, n=int(op.args.get("n") or 2), max_render_frames=max_render_frames
            )
        else:
            r["result"] = {"skipped": True}
        results.append(r)
        if r.get("failed"):
            break
    return {"ok": not any(x.get("failed") for x in results), "steps": results, "bindings": list(bindings.keys())}
