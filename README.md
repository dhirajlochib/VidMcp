<p align="center">
  <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/01_flowfield.gif" width="640" alt="VidMCP flow field" />
</p>

<h1 align="center">VidMCP</h1>

<p align="center">
  <em>Agents don’t describe video. They cut it.</em>
</p>

<p align="center">
  <a href="https://vidmcp.com"><img alt="Website" src="https://img.shields.io/badge/vidmcp.com-d4ff2a?style=flat-square&labelColor=050507&color=d4ff2a" /></a>
  <a href="https://dhirajlochib.com/"><img alt="Author" src="https://img.shields.io/badge/Dhiraj%20Lochib-2affd1?style=flat-square&labelColor=050507&color=2affd1" /></a>
  <a href="https://github.com/dhirajlochib/VidMcp"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-b18cff?style=flat-square&labelColor=050507&color=b18cff" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python_3.11+-ffa24e?style=flat-square&labelColor=050507" />
  <img alt="License" src="https://img.shields.io/badge/MIT-efefeb?style=flat-square&labelColor=050507" />
</p>

<p align="center">
  <b>MCP server</b> for text-prompt mattes · behind-the-subject VFX · education scenes<br/>
  Built by <a href="https://dhirajlochib.com/">Dhiraj Lochib</a> · Site <a href="https://vidmcp.com">vidmcp.com</a>
</p>

---

## Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/03_behind_subject.gif" width="520" alt="Behind the subject" /><br/>
  <sub>Behind the subject — particles under the matte</sub>
</p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/02_tesseract.gif" width="100%" alt="Tesseract" /><br/>
      <sub><b>Tesseract</b> · 4D with trails</sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/04_kinetic.gif" width="100%" alt="Kinetic" /><br/>
      <sub><b>Kinetic</b> · agent edit language</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/05_unit_circle.gif" width="100%" alt="Unit circle" /><br/>
      <sub><b>Unit circle</b> · live sin / cos</sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/06_pipeline.gif" width="100%" alt="Pipeline" /><br/>
      <sub><b>Pipeline</b> · import → render</sub>
    </td>
  </tr>
</table>

| Sample | GIF | MP4 |
|--------|-----|-----|
| Flow field | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/01_flowfield.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/01_flowfield.mp4) |
| Tesseract | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/02_tesseract.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/02_tesseract.mp4) |
| Behind the subject | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/03_behind_subject.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/03_behind_subject.mp4) |
| Kinetic | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/04_kinetic.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/04_kinetic.mp4) |
| Unit circle | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/05_unit_circle.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/05_unit_circle.mp4) |
| Pipeline | [gif](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/06_pipeline.gif) | [mp4](https://raw.githubusercontent.com/dhirajlochib/VidMcp/main/demos/samples/06_pipeline.mp4) |

```bash
python scripts/generate_samples.py
```

---

## Install

**Needs:** Python 3.11+, [ffmpeg](https://ffmpeg.org/)

```bash
git clone https://github.com/dhirajlochib/VidMcp.git
cd VidMcp
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

export VIDMCP_SAM_BACKEND=mock
export VIDMCP_WORKSPACE_ROOT=./workspaces
vidmcp --doctor
vidmcp
```

### Optional

```bash
pip install -e ".[mlx]"   # Apple Silicon SAM 3.1
pip install -e ".[sam]"   # CUDA
pip install -e ".[dev]"   # tests
```

---

## MCP (Claude Desktop)

```json
{
  "mcpServers": {
    "vidmcp": {
      "command": "/ABS/PATH/VidMcp/.venv/bin/vidmcp",
      "env": {
        "VIDMCP_SAM_BACKEND": "mock",
        "VIDMCP_WORKSPACE_ROOT": "/ABS/PATH/VidMcp/workspaces"
      }
    }
  }
}
```

```bash
claude mcp add vidmcp -s user -- /ABS/PATH/VidMcp/.venv/bin/vidmcp
```

---

## What it does

| | |
|---|---|
| **Segment** | Text-prompt subject matte (SAM / MLX / mock) |
| **Compose** | Effects *behind* the subject |
| **Educate** | Math plates + narration hooks |
| **Harness** | Plan → perceive → critic gates |

```
create_project → import_video → segment_subject
  → apply_background_effects → composite_and_render
```

---

## Project

```text
src/vidmcp/       MCP server + tools
demos/samples/    GIFs · MP4s · stills
site/             vidmcp.com (Hostinger)
configs/          Claude / Cursor
scripts/          generate_samples.py
```

---

## Author

**Dhiraj Lochib** — Full-Stack, Blockchain, AI  
[dhirajlochib.com](https://dhirajlochib.com/) · [GitHub](https://github.com/dhirajlochib) · [LinkedIn](https://www.linkedin.com/in/dhirajlochib-dev/) · dhirajch145@gmail.com

---

## License

[MIT](LICENSE)
