# Marketing guide — VidMCP (Dhiraj Lochib)

## Positioning (use this language)

**One line**
> Video tools your agent can actually use.

**Not this**
> Neon AI cyberpunk future video magic

**Yes this**
> Calm, open-source MCP for segmentation, behind-the-subject composition, and education scenes.

## Brand feel
- Warm dark UI, gold accent, serif headlines
- Short demos (6–7s), soft motion, no glitch spam
- Site: `site/` → Hostinger `public_html`
- Domain: **vidmcp.com**

## Assets to post
| Asset | Path | Use |
|-------|------|-----|
| Hero GIF | `demos/samples/01_tesseract_4d.gif` | README, X |
| Product GIF | `demos/samples/05_behind_subject.gif` | “the idea” |
| Kinetic | `demos/samples/02_kinetic.gif` | product loop |
| MP4 masters | `demos/samples/*.mp4` | LinkedIn / PH |

## Channels (order that works)

### 1) Day 0 — live
- [ ] Hostinger: upload `site/` → SSL → https://vidmcp.com
- [ ] Email: `hello@vidmcp.com`
- [ ] GitHub homepage = https://vidmcp.com
- [ ] Topics: `mcp` `video` `ai` `sam` `education` `vfx`

```bash
gh repo edit dhirajlochib/VidMcp \
  --homepage "https://vidmcp.com" \
  --add-topic mcp --add-topic video --add-topic ai \
  --add-topic education --add-topic sam --add-topic vfx
```

### 2) Social posts (copy)

**X / Twitter**
> Agents can write code. Video was still manual.
>
> VidMCP: text-prompt mattes, effects *behind* the subject, education plates — as MCP tools for Claude & Cursor.
>
> 🌐 vidmcp.com
> ★ github.com/dhirajlochib/VidMcp
>
> (attach: behind_subject or tesseract mp4)

**LinkedIn**
> Shipping VidMCP — open-source MCP so agents can segment and composite video, not just describe it.
>
> Built for real workflows: mattes under the speaker, math lessons with structure, quality-gated pipelines.
>
> Site: vidmcp.com · Code on GitHub.

**Show HN**
> Show HN: VidMCP – MCP server for agent-driven video editing (segment, compose, educate)

### 3) Community (don’t spam)
- r/ClaudeAI, r/LocalLLaMA — show GIF + github, answer questions
- Cursor / Claude Discords — #showcase
- MCP lists: awesome-mcp-servers, Glama, PulseMCP, Smithery
- Official MCP Registry after PyPI (`server.json`)

### 4) Product Hunt (week 2+)
- Tagline: *AI video tools for agents, via MCP*
- Gallery: 3 GIFs + site screenshot
- First comment: origin story + install + mock backend honesty

### 5) Content flywheel
- Weekly: one 10s plate from real use (education / creator)
- Thread: “How behind-the-subject VFX works in MCP”
- Short YouTube: install Claude Desktop in 90s

## Metrics that matter early
1. GitHub stars + unique clones
2. Site visits (Hostinger analytics / Cloudflare)
3. Issues / “I tried it” replies
4. MCP client installs (anecdotal)

## What not to do
- Fake “Y Combinator / funded” vibes
- Neon overload on social
- Promise Hollywood VFX without SAM weights
- Ignore mock-backend truth for first install

## Hostinger re-upload after redesign
1. Zip `site/` contents
2. Upload to `public_html/`
3. Hard refresh / CDN purge
4. Confirm https://vidmcp.com/assets/demos/01_tesseract_4d.mp4 plays
