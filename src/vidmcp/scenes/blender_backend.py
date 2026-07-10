"""Optional Blender headless scene renderer.

If `blender` binary exists, runs a generated Python script; otherwise returns None
so callers fall back to Manim/procedural.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.blender")


def blender_available() -> bool:
    return shutil.which("blender") is not None


def build_blender_script(prompt: str, out_mp4: Path, *, frames: int = 48, fps: int = 24) -> str:
    safe = prompt.replace("\\", "\\\\").replace('"', "'")[:120]
    out_mp4 = Path(out_mp4)
    # Blender writes frame sequence or ffmpeg; use image sequence then note path
    out_pattern = str(out_mp4.with_suffix("").as_posix()) + "_####.png"
    return textwrap.dedent(
        f'''
        import bpy
        import math

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.types, "BLENDER_EEVEE_NEXT") else "BLENDER_EEVEE"
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
        scene.render.fps = {fps}
        scene.frame_start = 1
        scene.frame_end = {frames}
        scene.render.filepath = r"{out_pattern}"
        scene.render.image_settings.file_format = "PNG"

        # camera
        bpy.ops.object.camera_add(location=(0, -6, 2))
        cam = bpy.context.object
        cam.rotation_euler = (math.radians(75), 0, 0)
        scene.camera = cam

        # light
        bpy.ops.object.light_add(type="AREA", location=(2, -2, 5))
        bpy.context.object.data.energy = 500

        # objects: icosphere + torus as abstract "math" sculpture
        bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), radius=1.0)
        sph = bpy.context.object
        bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0), major_radius=1.6, minor_radius=0.15)
        tor = bpy.context.object

        # simple emission material
        mat = bpy.data.materials.new("Neon")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        em = nodes.new("ShaderNodeEmission")
        em.inputs[0].default_value = (0.2, 0.6, 1.0, 1)
        em.inputs[1].default_value = 20
        mat.node_tree.links.new(em.outputs[0], out.inputs[0])
        sph.data.materials.append(mat)
        tor.data.materials.append(mat)

        # animate rotation
        for f in range(1, {frames} + 1):
            t = f / {frames}
            sph.rotation_euler = (t * math.tau, t * math.tau * 0.5, 0)
            sph.keyframe_insert(data_path="rotation_euler", frame=f)
            tor.rotation_euler = (0, 0, t * math.tau)
            tor.keyframe_insert(data_path="rotation_euler", frame=f)

        # title empty note stored in text object if available
        bpy.ops.object.text_add(location=(-2.5, 0, 2.2))
        txt = bpy.context.object
        txt.data.body = "{safe}"
        txt.scale = (0.35, 0.35, 0.35)

        bpy.ops.render.render(animation=True)
        print("VIDMCP_BLENDER_DONE")
        '''
    ).strip() + "\n"


def render_blender_scene(
    prompt: str,
    *,
    out_dir: Path,
    frames: int = 36,
    fps: int = 24,
    timeout_sec: float = 300.0,
) -> Path | None:
    if not blender_available():
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "blender_scene.py"
    out_mp4 = out_dir / "blender_render.mp4"
    script.write_text(build_blender_script(prompt, out_mp4, frames=frames, fps=fps), encoding="utf-8")
    cmd = ["blender", "-b", "-P", str(script)]
    log.info("blender_render_start", cmd=" ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except Exception as e:  # noqa: BLE001
        log.warning("blender_failed", error=str(e))
        return None
    if proc.returncode != 0:
        log.warning("blender_nonzero", stderr=(proc.stderr or "")[-500:])
        # still try to assemble frames
    # stitch pngs
    pngs = sorted(out_dir.glob("blender_render_*.png")) + sorted(out_dir.glob("*_####.png"))
    # generated pattern blender_render_0001.png style from filepath
    pngs = sorted(out_dir.glob("blender_render_*.png"))
    if not pngs:
        # try any numbered pngs
        pngs = sorted(out_dir.glob("*.png"))
    if not pngs:
        return None
    # ffmpeg stitch
    try:
        # rename to sequential if needed
        seq_dir = out_dir / "seq"
        seq_dir.mkdir(exist_ok=True)
        for i, p in enumerate(pngs):
            target = seq_dir / f"frame_{i:04d}.png"
            if not target.exists():
                shutil_copy = __import__("shutil").copy2
                shutil_copy(p, target)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(seq_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(out_mp4),
            ],
            check=True,
            capture_output=True,
        )
        return out_mp4 if out_mp4.exists() else None
    except Exception as e:  # noqa: BLE001
        log.warning("blender_stitch_failed", error=str(e))
        return None
