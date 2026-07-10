<p align="center">
  <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/07_creator_polish.gif" width="640" alt="VidMCP 1.1 Creator polish" />
</p>

<h1 align="center">VidMCP</h1>

<p align="center">
  <em>Agents don’t describe video. They cut it.</em>
</p>

<p align="center">
  <a href="https://vidmcp.com"><img alt="Website" src="https://img.shields.io/badge/vidmcp.com-d4ff2a?style=flat-square&labelColor=050507&color=d4ff2a" /></a>
  <a href="https://pypi.org/project/vidmcp/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-2affd1?style=flat-square&labelColor=050507" /></a>
  <a href="https://dhirajlochib.com/"><img alt="Author" src="https://img.shields.io/badge/Dhiraj%20Lochib-b18cff?style=flat-square&labelColor=050507" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python_3.11+-ffa24e?style=flat-square&labelColor=050507" />
  <img alt="License" src="https://img.shields.io/badge/MIT-efefeb?style=flat-square&labelColor=050507" />
</p>

<p align="center">
  <b>MCP server</b> for agent video editing — mattes · behind-the-subject VFX · audio polish · captions · export<br/>
  Built by <a href="https://dhirajlochib.com/">Dhiraj Lochib</a> · <a href="https://vidmcp.com">vidmcp.com</a> · <code>uv tool install vidmcp</code>
</p>

---

## Creator polish pipeline

One call for a publish-ready talk-head:

```text
orient → denoise → BGM → captions → optional BG replace → export preset
```

| Tool | What it does |
|------|----------------|
| `run_talking_head_polish` | Full recipe in one shot |
| `process_audio` | Denoise · gate · EQ · loudnorm |
| `mix_bgm` | Ambient pad under voice (ducking) |
| `transcribe_and_caption` | Whisper words + brand burn-in |
| `replace_background` | Matte + space / blur / solid plate |
| `smart_cut_hesitations` | Dead air & filler removal |
| `export_render` | `youtube_16x9` · `reels_9x16` · `square_1x1` |
| `import_video` | **Bakes portrait rotation** upright by default |

```python
from vidmcp.tools.creator import run_talking_head_polish

print(run_talking_head_polish(
    "talk.mov",
    preset="reels_9x16",
    bg_mode="space",   # none | space | blur | solid
    strength=0.75,
    bgm_volume=0.35,
))
```

MCP agents can also call `apply_recipe(..., recipe_name="talking_head_polish")` or `list_tool_packs()`.

---

## Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/07_creator_polish.gif" width="560" alt="Creator 1.1" /><br/>
  <sub><b>1.1 Creator polish</b> — orient · denoise · BGM · captions · export</sub>
</p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/03_behind_subject.gif" width="100%" alt="Behind subject" /><br/>
      <sub><b>Behind the subject</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/02_tesseract.gif" width="100%" alt="Tesseract" /><br/>
      <sub><b>Tesseract</b> · 4D trails</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/01_flowfield.gif" width="100%" alt="Flow" /><br/>
      <sub><b>Flow field</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/05_unit_circle.gif" width="100%" alt="Unit circle" /><br/>
      <sub><b>Unit circle</b></sub>
    </td>
  </tr>
</table>

| Sample | GIF | MP4 |
|--------|-----|-----|
| **Creator polish** | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/07_creator_polish.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/07_creator_polish.mp4) |
| Flow field | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/01_flowfield.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/01_flowfield.mp4) |
| Tesseract | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/02_tesseract.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/02_tesseract.mp4) |
| Behind the subject | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/03_behind_subject.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/03_behind_subject.mp4) |
| Kinetic | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/04_kinetic.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/04_kinetic.mp4) |
| Unit circle | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/05_unit_circle.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/05_unit_circle.mp4) |
| Pipeline | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/06_pipeline.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/06_pipeline.mp4) |

---

## Install

**Needs:** Python 3.11+, [ffmpeg](https://ffmpeg.org/), [uv](https://docs.astral.sh/uv/) recommended

```bash
uv tool install vidmcp
vidmcp --doctor

# or
pip install 'vidmcp[creator]'   # + faster-whisper + mediapipe
```

### MCP (Grok / Claude / Cursor)

```bash
# Grok
grok mcp add vidmcp \
  -e VIDMCP_SAM_BACKEND=mock \
  -e VIDMCP_WORKSPACE_ROOT=$HOME/vidmcp-workspaces \
  -- uvx vidmcp
```

```json
{
  "mcpServers": {
    "vidmcp": {
      "command": "uvx",
      "args": ["vidmcp"],
      "env": {
        "VIDMCP_SAM_BACKEND": "mock",
        "VIDMCP_WORKSPACE_ROOT": "/path/to/workspaces"
      }
    }
  }
}
```

Optional real matte on Apple Silicon:

```bash
pip install 'vidmcp[mlx]'
export VIDMCP_SAM_BACKEND=mlx
export VIDMCP_MLX_MODEL_ID=mlx-community/sam3.1-bf16
```

---

## Core product

| | |
|---|---|
| **Segment** | Text-prompt subject matte (SAM / MLX / MediaPipe / mock) |
| **Compose** | Effects *behind* the subject |
| **Audio** | Denoise, enhance, BGM duck, loudnorm |
| **Captions** | Whisper timeline + brand burn-in |
| **Educate** | Math plates + speech-locked scenes |
| **Export** | 16:9 · 9:16 · 1:1 without stretch |

```text
create_project → import_video (orient)
  → process_audio → mix_bgm
  → segment_subject | replace_background
  → transcribe_and_caption → export_render
```

Prefer high-level paths: `run_talking_head_polish` · `apply_recipe` · `run_quality_gated_pipeline`.

---

## Project layout

```text
src/vidmcp/     MCP server, creator tools, perception, effects
demos/samples/  GIFs · MP4s for README & site
site/           vidmcp.com (static Hostinger)
configs/        Claude / Cursor snippets
```

---

## Author

**Dhiraj Lochib** — Full-Stack, Blockchain, AI  
[dhirajlochib.com](https://dhirajlochib.com/) · [GitHub](https://github.com/dhirajlochib) · [LinkedIn](https://www.linkedin.com/in/dhirajlochib-dev/) · dhirajch145@gmail.com

---

## License

[MIT](LICENSE)
