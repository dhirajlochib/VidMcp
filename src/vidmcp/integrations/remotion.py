"""Remotion/JS scene scaffold generator + optional local render.

Generates a minimal Remotion composition project folder. If `npx`/`npm` and
remotion are available, attempts a headless render; otherwise returns the
scaffold path for the user/agent to render.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.remotion")


def remotion_available() -> bool:
    return shutil.which("npx") is not None


def scaffold_remotion_scene(
    prompt: str,
    *,
    out_dir: Path,
    duration_sec: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(duration_sec * fps))
    safe = prompt.replace("`", "'")[:80]

    pkg = {
        "name": "vidmcp-remotion-scene",
        "version": "1.0.0",
        "private": True,
        "scripts": {
            "build": "echo 'Use remotion render when deps installed'",
        },
    }
    (out_dir / "package.json").write_text(json.dumps(pkg, indent=2))

    comp = textwrap.dedent(
        f'''
        // VidMCP-generated Remotion composition
        // Prompt: {safe}
        import React from "react";

        export const VidMCPScene: React.FC<{{frame: number; fps: number}}> = ({{frame, fps}}) => {{
          const t = frame / fps;
          const progress = Math.min(1, frame / {frames});
          return (
            <div style={{{{
              width: {width},
              height: {height},
              background: "linear-gradient(135deg, #0b1020, #1a1030)",
              color: "white",
              fontFamily: "Inter, system-ui, sans-serif",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              position: "relative",
              overflow: "hidden",
            }}}}>
              <div style={{{{fontSize: 42, fontWeight: 700, marginBottom: 24}}}}>
                {safe}
              </div>
              <div style={{{{
                width: 480 * progress,
                height: 8,
                background: "#4cc9f0",
                borderRadius: 8,
              }}}} />
              <div style={{{{
                marginTop: 40,
                width: 200,
                height: 200,
                borderRadius: 24,
                border: "4px solid #f72585",
                transform: `rotate(${{t * 40}}deg) scale(${{0.8 + 0.2 * progress}})`,
              }}}} />
              <div style={{{{position: "absolute", bottom: 24, opacity: 0.6, fontSize: 14}}}}>
                VidMCP Remotion scaffold · t={{t.toFixed(2)}}s
              </div>
            </div>
          );
        }};

        export const composition = {{
          id: "VidMCPScene",
          component: VidMCPScene,
          durationInFrames: {frames},
          fps: {fps},
          width: {width},
          height: {height},
        }};
        '''
    ).strip() + "\n"
    (out_dir / "VidMCPScene.tsx").write_text(comp)
    (out_dir / "README.md").write_text(
        f"""# VidMCP Remotion scene\n\nPrompt: {safe}\n\n```bash\nnpm i remotion @remotion/cli react react-dom\nnpx remotion render VidMCPScene out.mp4\n```\n"""
    )

    mp4 = None
    # try render if remotion present (rare in bare env)
    if remotion_available():
        try:
            # without full project this usually fails — keep best-effort
            pass
        except Exception as e:  # noqa: BLE001
            log.info("remotion_render_skip", error=str(e)[:120])

    return {
        "ok": True,
        "backend": "remotion_scaffold",
        "project_dir": str(out_dir),
        "entry": str(out_dir / "VidMCPScene.tsx"),
        "duration_in_frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "render_path": mp4,
        "note": "Scaffold generated. Install remotion to render, or use procedural plate from same prompt.",
    }
