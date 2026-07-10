# VidMCP — God-Level Upgrade Roadmap

> **STATUS (2026-07-11): IMPLEMENTED.** All ten workstreams landed in one pass — 40+ new
> modules, ~45 new MCP tools (144 total), 5 new tool packs (`director`, `editor`, `colorist`,
> `sound`, `mograph`), 7 v2 recipes, 33 new unit tests (85 total, all passing), end-to-end
> smoke verified on a synthetic clip. Every feature ships with its dependency-free fallback;
> model-backed upgrades activate via `ensure_model`. Remaining open slots (deliberate stubs):
> XTTS cloning adapter, MusicGen/provider music, neural style transfer onnx, Remotion render,
> MatAnyone weights wiring — each has a working fallback behind the same tool.

**Single-pass execution document.** This is not a phased plan. It is one dependency-ordered
document an AI agent (or human) can work through top-to-bottom in a single sustained effort.
Workstreams are ordered so that everything a later section needs already exists by the time
you reach it. Every workstream states: what exists today (with file references), what to
build, the exact new MCP tool surface, the models/dependencies involved, implementation
notes, and measurable acceptance gates.

**Version target:** VidMCP 2.0 — "the agent doesn't just apply effects; it *edits*."

---

## 0. North Star — Definition of a God-Level Edit

The test scenario that defines done. An agent is given one raw 12-minute talking-head +
screen-capture recording and the intent *"make this a tight, cinematic 8-minute YouTube
video plus a 45-second vertical teaser"*. With **zero human intervention** it must:

1. **Understand** the footage: shots, scenes, who is speaking when, what is said (word-level),
   where the energy/emotion peaks are, where the mistakes/retakes/fillers are, which pauses
   are dramatic vs dead.
2. **Plan** a full edit — narrative arc, pacing curve, cut list, B-roll insert points,
   graphics beats, music mood — and expose that plan as an inspectable, revisable artifact
   *before* rendering anything.
3. **Cut** it: remove retakes/fillers while preserving meaningful pauses, fill room tone
   under cuts, punch in on emphasis, cut B-roll on beat.
4. **Look** god-level: hair-accurate subject matte with zero flicker, behind-subject
   graphics correctly occluded, matched lighting, real color grade (LUT + auto white balance
   + shot-to-shot match), smart reframe to 9:16 without stretching, motion graphics from the
   creator's brand kit.
5. **Sound** god-level: isolated clean voice, platform-target loudness, music that matches
   the pacing curve and ducks under speech, SFX on graphic hits, no cut clicks.
6. **Check itself**: re-watch the render (visual + audio QC), catch bad cuts / sync drift /
   matte failures / caption typos, and auto-repair before declaring done.
7. **Deliver**: one call produces YouTube 16:9 + Reels 9:16 + square, each with correct
   codec/bitrate/loudness, plus 3 A/B thumbnails, SRT sidecars, chapters, title/description.

Every feature below exists to close a gap between the current codebase and that scenario.

---

## 1. Gap Audit — Current State vs Target

| Area | Today (files) | Gap to god-level |
|---|---|---|
| Matte | SAM 3/3.1 backends (`perception/*`), MediaPipe fallback (`matte/fast_matte.py`), keyframe refine, uncertainty field (`advanced/uncertainty.py`) | Binary-ish masks, no hair-level alpha, flicker suppression is heuristic heal, no light wrap/spill/shadow |
| Color | `effects/grading.py` = 28 lines of contrast/sat/temp; `advanced/lighting_match.py` Lab transfer | No LUTs, no auto WB/exposure, no shot match, no scopes, no skin protection |
| Camera | `media/export.py` pads/crops statically | No saliency reframe, no Ken Burns, no punch-ins, no speed ramps, no stabilization |
| Understanding | histogram shots (`advanced/shot_detect.py`), energy diarization fallback (`audio/diarize.py`), Whisper words (`audio/whisper_timeline.py`) | No scene clustering, no emotion/action/laughter detection, no semantic search, no B-roll matching |
| Cutting | `edit/smart_cut.py` gap+filler heuristics | No retake detection, no pause semantics, no room-tone fill, no beat-aligned cuts |
| Audio | afftdn chain (`audio/process.py`), procedural pads (`audio/bgm.py`), simple duck | No stem separation, no multi-track model, no music gen, no dubbing/cloning, single LUFS target |
| MoGraph | particles/flowfield (`effects/particles.py`), infographic cards (`edit/infographics.py`), Manim/procedural scenes (`scenes/*`) | No lower thirds/titles/charts DSL, no brand kit, no style transfer, Remotion is scaffold-only |
| Recipes | 10 static dicts (`harness/recipes.py`) | No schema, no composition/inheritance, no adaptivity, no brand injection |
| Agents | regex planner (`agents/planner.py`), metric critic (`agents/critic.py`, `critics/ensemble.py`), gates (`harness/quality_gates.py`) | No persisted executable plan, no VLM re-watch, no creative-intent reasoning, no batched op execution |
| Delivery | 3 presets, CRF 18 (`media/export.py`), single thumbnail | No multi-target batch, no codec/bitrate ladder, no thumbnail A/B, no metadata pack |

Note: `scripts/edit_time_dilation_v2.py` and `scripts/edit_dimensions_pro.py` are hand-rolled
prototypes of exactly what W6 (premium graphics/HUD) and W5 (pro vocal chain + duck) should
productize. Mine them for the visual language; then delete them once the features land as tools.

---

## 2. Execution Order (Dependency Graph)

Work top-to-bottom. Arrows = "consumed by".

```
W10 Infra (model manager, frame cache, op batching)      ──► everything
W1  Matte & Compositing Core                              ──► W2, W3, W6
W4  Video Understanding & Semantic Index                  ──► W3, W5, W7, W8
W2  Color Science                                         ──► W6, W8-QC
W5  Audio Suite                                           ──► W8-QC, W9
W3  Virtual Camera / Reframe / Time                       ──► W9
W6  Motion Graphics, Brand, Style                         ──► W7
W7  Recipe System v2 + Adaptive Templates                 ──► W8
W8  Agent Cognition (plan → execute → self-critique)      ──► W9
W9  Delivery & Distribution                               ──► done
```

W10 comes first because every later workstream produces heavier per-frame work; without the
frame cache, op batching, and model manager the rest is too slow to iterate on.

---

## W10. Infrastructure Foundation (do first)

### W10.1 Unified model manager
**Today:** only SAM weights have a resolver (`perception/weights.py`, `ensure_sam_weights` tool).
**Build:** `vidmcp/models_registry.py` — declarative registry of every optional model this
roadmap adds (matting, depth, saliency, audio tagging, stem separation, ASR, TTS, frame
interpolation, embeddings). Each entry: name, source (HF repo / URL), size, license, device
requirements, lazy download, integrity hash, cache path under `~/.cache/vidmcp/models`.

```
Tool: ensure_model(name: str, download: bool = False) -> {found, path, size_mb, license, device_ok}
Tool: list_models() -> registry with installed/missing status
```
Generalizes `ensure_sam_weights`; keep that tool as an alias. `platform_health` reports the
whole registry.

### W10.2 Content-addressed frame cache + proxy pipeline
**Today:** every op re-decodes the source; `harness_preview_frames=48` is the only speed lever.
**Build:** `vidmcp/core/framecache.py`:
- Decode once to a **mezzanine** (ProRes-proxy-style or high-CRF h264 + PNG mask dirs already exist).
- Cache key = SHA of (input content hash, op name, canonical params). Op outputs (mask dirs,
  graded frame dirs, plates) are stored content-addressed in `project/cache/` and reused —
  re-running a recipe with one changed param only recomputes downstream of the change.
- **Proxy workflow:** all analysis and preview passes run at ≤540p proxies
  (`settings.proxy_max_side`, default 540); only the final composite touches full res. Tools
  gain `quality: "proxy"|"final"` defaulting to proxy; `composite_and_render` defaults final.
- macOS: use VideoToolbox (`h264_videotoolbox`/`hevc_videotoolbox`) for encode when available
  (`ffmpeg_ops.py` gets an `hw_encoder()` probe).

### W10.3 Server-side op batching (agent call efficiency)
**Today:** agents burn one MCP round-trip per operation; `enhance_edit` is a hardcoded batch.
**Build:** generic executor:
```
Tool: run_ops(project_id, ops: list[{tool, args, id?, if?: condition}], stop_on_error=False)
      -> {results: {id: compact_result}, timeline_ms, failures}
```
- `if` conditions reference prior op results (`"if": "seg.temporal_stability < 0.65"`), giving
  agents server-side branching without extra round-trips.
- Every result passes through `utils/compact.py`. This tool alone should cut typical agent
  token usage 3–5× on multi-step edits. Deprecate `enhance_edit`'s hardcoded chain in favor of
  a shipped op-list preset.

### W10.4 Async job semantics for heavy tools
**Today:** `core/job_manager.py` + file queue (`queue/worker.py`) exist but most tools block.
**Build:** every tool that can exceed ~20s gains `async_mode: bool = False`; when true returns
`job_id` immediately and the agent polls `get_job_status` (already exists). Wire the durable
queue handlers for all new heavy ops (stem separation, frame interpolation, batch export).

**Acceptance gates (W10):** re-running an unchanged recipe = >90% cache hits, <10% of original
wall time. `run_ops` executes a 6-op chain in one MCP call with conditional branch taken
correctly. All new models resolvable via `ensure_model` with no import-time downloads.

---

## W1. Matte & Compositing Core

The matte is the product's spine. Two upgrades: **alpha quality** and **temporal stability**,
then compositing realism on top.

### W1.1 Hair-level alpha refinement stage
**Today:** SAM masks are semantic-level; `_soft()` in `matte/fast_matte.py` blurs edges — that
is feathering, not matting. Hair, motion blur, semi-transparency all fail.
**Build:** new module `vidmcp/matte/alpha_refine.py` — a post-pass that converts any binary
mask track into a true alpha track:
1. Generate trimap per frame from the SAM mask (erode → definite FG, dilate → definite BG,
   band = unknown; band width scales with `uncertainty` field from `advanced/uncertainty.py`).
2. Run a video matting model over the unknown band only:
   - **Primary:** MatAnyone (memory-based video matting, consistent by design).
   - **Fallback A:** RobustVideoMatting (RVM) — fast, no trimap, good for talking heads.
   - **Fallback B:** ViTMatte on keyframes + flow-propagated alpha between keyframes.
   - **CPU fallback:** guided filter (`cv2.ximgproc.guidedFilter`) alpha snap — always works.
3. Store as 16-bit PNG alpha dir alongside the mask dir; `SegmentTrack` gains `alpha_dir`.

```
Tool: refine_alpha(project_id, segment_id?, backend="auto", band_px=16) -> {alpha_dir, backend, edge_quality}
```
Compositor (`compositor/engine.py`, `compositor/alpha.py`) prefers `alpha_dir` when present.

### W1.2 Temporal consistency engine
**Today:** `refine_segment_keyframes` heals bad frames; `temporal_stability_score` measures
IoU drift. Nothing *enforces* stability during compositing.
**Build:** `vidmcp/matte/temporal.py`:
- Optical-flow-guided alpha warping: warp α(t-1) by flow, blend with α(t) weighted by per-pixel
  uncertainty (flicker pixels trust the warp; confident pixels trust the fresh estimate).
- Edge-band EMA with scene-cut reset (consume W4 shot boundaries when available; histogram
  fallback exists in `advanced/shot_detect.py`).
- New metrics in `perception/mask_ops.py`: **dtSSD** (temporal alpha gradient error) and an
  edge-flicker index; both feed `quality_gates.py` and `matte_diagnostics`.

```
Tool: stabilize_matte(project_id, strength=0.6) -> {dtSSD_before, dtSSD_after, frames_touched}
```

### W1.3 Compositing realism pass
**Today:** `advanced/lighting_match.py` (Lab transfer) and depth fog exist. Composites still
read as stickers in hard cases.
**Build in `compositor/engine.py` as optional per-composite flags:**
- **Light wrap:** blur the BG plate, screen-blend into a thin dilated edge band of the subject.
- **Spill/color decontamination:** suppress BG-dominant hue in the alpha edge band.
- **Contact shadow:** project the matte silhouette (squashed, blurred, opacity from BG
  luminance) onto the plate below the subject's lowest matte point; use depth field when
  available to place it.
- **Grain/noise match:** estimate source grain σ (high-pass std on flat regions), add matched
  grain to synthetic plates so BG doesn't look "too clean".
- **Depth-ordered layers:** generalize the fog z-logic (`depth/fog.py`) — every layer gets an
  optional `z` so graphics can pass in front/behind per-region using the depth field.

```
Tool: composite_and_render(..., realism: {light_wrap, decontaminate, contact_shadow, grain_match} = all-on-final)
```

### W1.4 Person-part mattes
**Build:** `segment_subject(prompt="person", parts=True)` → also emit hair/face/torso/hands
sub-mattes (SAM 3 sub-prompts, or MediaPipe face mesh for the face region). Consumers: W2
skin-tone protection, W3 face-locked reframe, W6 graphics that tuck behind a shoulder.

**Acceptance gates (W1):** alpha edge SAD vs guided-filter baseline improves ≥30% on the demo
set; dtSSD after `stabilize_matte` ≤ 0.5× before; composite over a bright plate shows no dark
halo (edge-band mean ΔE < 4 vs subject interior); gates get `min_alpha_quality` and
`max_edge_flicker` thresholds in `config.py`.

---

## W4. Video Understanding & Semantic Index

(Ordered before W2/W3/W5 because reframe, smart cut, music, and adaptive recipes all consume it.)

### W4.1 Shot + scene segmentation upgrade
**Today:** histogram threshold in `advanced/shot_detect.py`.
**Build:** `vidmcp/perception/scene_seg.py`:
- **TransNetV2** (via `ensure_model`) for shot boundaries; PySceneDetect adaptive detector as
  dependency-light fallback; histogram as last resort.
- Cluster shots into **scenes** by CLIP-embedding similarity of shot keyframes + temporal
  adjacency. Persist to manifest: `manifest.analysis.shots = [{start, end, keyframe, scene_id}]`.

```
Tool: detect_scenes(project_id, backend="auto") -> {n_shots, n_scenes, shots[...compact]}
```
`detect_shots` becomes an alias.

### W4.2 The Footage Index — one pass, everything extracted
**Build:** `vidmcp/perception/indexer.py` — a single orchestrated analysis pass per source
that writes `project/index/footage_index.json` (+ embedding matrices as `.npy`):
- **Visual:** per-shot keyframes → SigLIP/CLIP image embeddings; aesthetic score (LAION
  aesthetic predictor head — tiny); face count/size/expression (MediaPipe face mesh + FER
  head); motion energy (existing `motion/optical_flow.py`); sharpness/exposure stats.
- **Audio events:** PANNs/AST audio tagging on 1s windows — laughter, applause, music, typing,
  silence classes.
- **Speech:** word timeline (existing Whisper path; upgrade default to `faster-whisper
  large-v3-turbo` when installed, WhisperX-style alignment optional), diarization (pyannote 3.1
  via `ensure_model`, existing spectral fallback), per-segment sentence embeddings
  (`sentence-transformers/all-MiniLM-L6-v2` — small, CPU-fine).
- **Emotion/energy curve:** fuse prosody (pitch/energy via librosa) + face expression +
  word sentiment into a per-second `energy[]` and `emotion[]` track.

```
Tool: build_footage_index(project_id, include=["visual","audio_events","speech","emotion"]) -> {index_path, stats}
```

### W4.3 Semantic search across footage
**Build:** `vidmcp/perception/search.py` over the index:
```
Tool: search_footage(project_id, query: str, modality="auto", top_k=5)
      -> [{t_start, t_end, score, why: "laughter_detected + transcript match 'that's hilarious'"}]
```
- Text query embedded once; scored against transcript embeddings, CLIP text→image embeddings,
  and audio-event tags simultaneously; fused ranking. "find the part where I laugh" hits the
  laughter audio event even if the transcript never says "laugh".
- Works across **multiple projects** too: `search_library(query)` scans all indexed projects —
  this is the B-roll retrieval backbone.

### W4.4 Semantic pauses, fillers, retakes (smart_cut v2)
**Today:** `edit/smart_cut.py` cuts on gap length + a static filler word list.
**Build:** `vidmcp/edit/cut_planner.py`:
- Classify every pause: **dead** (no prosodic contour, mid-clause), **breath**, **dramatic**
  (follows emphasis word / precedes topic shift / high preceding energy), **retake boundary**
  (following sentence is a near-duplicate embedding of the previous one → keep the LAST take).
- Filler detection from ASR word confidence + POS context ("like" as comparator survives,
  "like" as filler dies).
- Output is a **cut plan artifact** (list of keep-ranges with reasons), not an immediate render
  — the agent (W8) reviews it before applying.
- On apply: **room-tone fill** — sample 300ms of ambient noise from a silent region, cross-fade
  15ms at every cut so audio never clicks or goes digitally dead.

```
Tool: plan_cuts(project_id, target_ratio?: float, keep_dramatic_pauses=True) -> {cut_plan_id, kept_sec, removed_sec, ranges[...with reasons]}
Tool: apply_cut_plan(project_id, cut_plan_id, room_tone=True) -> {output, edl}
```
`smart_cut_hesitations` delegates to these with defaults.

### W4.5 Auto B-roll suggestion + matching
**Build:** `vidmcp/edit/broll_match.py`:
- For each transcript beat, query `search_library` + the current project's own non-A-roll
  footage for visually/semantically matching clips; score by embedding similarity × aesthetic
  score × motion compatibility.
- If no real footage matches: fall back to generated plates (W6.5) or stock-provider hook
  (Pexels/Pixabay API adapters behind `VIDMCP_STOCK_API_KEY`).
- Insert with J-cut audio lead defaults, duration snapped to the pacing grid (W8.2).

```
Tool: suggest_broll(project_id, beats?: auto) -> [{t, duration, source: clip|generated|stock, preview}]
Tool: insert_broll(project_id, suggestions: list, transition="cut") -> {layers_added}
```

**Acceptance gates (W4):** index build ≤ 1.5× realtime at proxy res on M-series; laughter
search returns correct segment on the demo set; retake detector catches a scripted double-take
fixture; cut plan keeps a hand-labeled dramatic pause fixture.

---

## W2. Color Science

### W2.1 LUT engine
**Build:** `vidmcp/color/lut.py`:
- Parse `.cube` (1D + 3D) LUTs; apply with tetrahedral interpolation (numpy vectorized;
  optional `colour-science` acceleration).
- Ship 8–10 original LUTs (teal-orange, filmic soft, noir, bleach bypass, warm portrait…)
  generated procedurally into `configs/luts/` — no copyright issues.
- LUT becomes an effect layer (`effect_type: "lut"`, params: `path|name, intensity`,
  `background_only` respected like existing grade).

```
Tool: apply_lut(project_id, lut: str, intensity=1.0, background_only=False) -> {layer_id}
Tool: list_luts() -> builtin + workspace/luts/*.cube
```

### W2.2 Auto white balance + exposure normalize
**Build:** `vidmcp/color/auto_correct.py`:
- WB: gray-world + white-patch fusion in Lab, gated by a skin-tone sanity check (skin pixels
  from W1.4 face matte must stay within CIE skin locus).
- Exposure: mid-gray anchoring on the subject matte region (subject correctly exposed even if
  the BG blows out); highlight roll-off instead of clip.
- Temporal smoothing of correction params across each shot (per-shot constants, eased at cuts).

```
Tool: auto_color(project_id, wb=True, exposure=True, per_shot=True) -> {per_shot_params, layer_id}
```

### W2.3 Shot-to-shot + reference match
**Today:** `advanced/lighting_match.py` does Lab mean/std transfer subject↔plate.
**Build:** extend to `vidmcp/color/match.py`:
- Match all shots to a chosen hero shot (or to a supplied reference image/video): 3D histogram
  matching in Lab with regularization (limit ΔE shift, preserve skin).
- "Grade like this" — extract a compact grade fingerprint (per-channel tone curves + sat) from
  any reference and apply as a layer.

```
Tool: match_color(project_id, reference: "shot:N" | path, strength=0.8) -> {layer_id, per_shot_delta}
```

### W2.4 Grade primitives + skin protection
**Build:** replace the 28-line `effects/grading.py` with `vidmcp/color/grade.py`:
lift/gamma/gain wheels, filmic tone curve, HSL secondary (hue-band select → shift), vibrance
(sat-aware saturation), and a **skin protection mask** (face matte dilated, excluded from
aggressive moves). Keep the old param names working.

### W2.5 Scopes for machines
**Build:** `vidmcp/color/scopes.py` — waveform, vectorscope, RGB histogram rendered to PNG +
returned as stats (clip %, skin-line deviation, cast estimate). Consumed by W8 self-critique
and available to the host LLM as images.

```
Tool: color_scopes(project_id, frame?: int|"auto") -> {waveform_png, vectorscope_png, stats}
```

**Acceptance gates (W2):** LUT apply bit-matches a reference implementation on a test ramp
within ±1/255; auto WB corrects a synthetically tinted fixture to <2 ΔE cast; skin ΔE < 5
after any auto grade; scopes stats flag a deliberately clipped fixture.

---

## W3. Virtual Camera — Reframe, Punch-ins, Time

### W3.1 Saliency-driven smart reframe (9:16 ⇄ 16:9 ⇄ 1:1, no stretch)
**Today:** `media/export.py` center pads/crops.
**Build:** `vidmcp/camera/reframe.py`:
- Saliency per frame = fused: subject matte centroid (W1) > face boxes (W1.4) > motion energy
  (existing optical flow) > center prior. No new heavy model needed initially; optional U²-Net
  saliency via `ensure_model` for non-person footage.
- Crop path solver: per-shot smoothing with a 1-Euro filter + max-velocity/acceleration
  constraints (virtual camera operator — pans only when the subject would exit a dead zone);
  hard reset at shot boundaries (W4.1). Two-subject scenes: prefer a crop containing the
  **active speaker** (diarization + face correlation from W4.2).
- Output is a keyframed crop track stored on the manifest → consumed by `export_render` and
  `export_multi` (W9), so one edit reframes to every aspect.

```
Tool: smart_reframe(project_id, target="9:16", mode="track_subject"|"track_speaker"|"static") -> {crop_track, preview}
```

### W3.2 Ken Burns + emphasis punch-ins
**Build:** `vidmcp/camera/moves.py`:
- Ken Burns for stills/plates: eased zoom/pan between two saliency-derived rects.
- **Emphasis punch-ins:** at high-emphasis words (energy peaks from W4.2 emotion curve, or
  keywords), digital punch 100%→110% with an ease-out over 8 frames; alternate in/out to
  create rhythm. This is the single highest-perceived-value "editor touch" for talking heads.

```
Tool: add_camera_moves(project_id, style="emphasis_punch"|"ken_burns"|"slow_drift", max_zoom=1.12) -> {n_moves, track}
```

### W3.3 Time warp — speed ramps, slow-mo, freeze
**Today:** prototype ideas live in `scripts/edit_time_dilation_v2.py`; nothing productized.
**Build:** `vidmcp/camera/timewarp.py`:
- Speed ramp spec: list of `{t, speed}` keyframes with eased interpolation; audio handled
  correctly — speech regions use pitch-preserving stretch (`rubberband` if installed, ffmpeg
  `atempo` chain fallback); music regions may resample.
- Smooth slow-mo: optical-flow frame blending (`minterpolate`) by default; **RIFE** frame
  interpolation via `ensure_model` for quality mode.
- Freeze frames with optional grade/graphic on the frozen beat.
- Guard: never time-warp inside a kept speech range unless `allow_speech_warp=True`.

```
Tool: time_warp(project_id, ramps: [{t, speed, ease}], quality="flow"|"rife"|"blend") -> {output_duration, layer}
```

### W3.4 Stabilization
**Build:** ffmpeg `vidstab` two-pass wrapped as `stabilize_video(project_id, strength)`;
run before segmentation when analysis detects handheld shake (motion energy high-freq
component). Crop-compensation aware of the reframe track (stabilize first, reframe second).

**Acceptance gates (W3):** reframe track never lets the face bbox leave the crop on the demo
set; crop velocity ≤ configured max (no jitter — verified by track second-derivative);
speech in a 2× ramped region stays intelligible + pitch-true; punch-ins land within 80ms of
emphasis peaks.

---

## W5. Audio Suite

### W5.1 Multi-track audio model
**Today:** audio is a single implicit chain (vocals_clean.wav + optional bgm mix).
**Build:** `vidmcp/audio/tracks.py` — manifest gains `audio_tracks`: named tracks
(`voice`, `bgm`, `sfx`, `ambience`, `dub_*`) each with clip list (src, t_in/out, gain,
fades) and an effect chain (eq, compress, gate, reverb-tail-kill). One mixdown function
renders the graph via a single ffmpeg `filter_complex` (build on `compositor/ffmpeg_ops.py`).

```
Tool: audio_tracks(project_id) -> compact track graph
Tool: edit_audio_track(project_id, track, ops: [{op, args}]) -> {track}
Tool: mixdown_audio(project_id, target="youtube") -> {wav, lufs, true_peak}
```

### W5.2 Voice isolation upgrade (stems)
**Today:** `audio/process.py` = afftdn + gate + EQ. Good, not god.
**Build:** optional **Demucs (htdemucs)** stem separation via `ensure_model` — split
voice/music/other; the voice stem replaces afftdn output when available (`process_audio`
gains `backend="auto"|"stems"|"filter"`). This also unlocks: strip badly-recorded room music
from source, keep only speech, remix with licensed BGM.

### W5.3 Ducking v2 + SFX
**Build:**
- VAD-driven sidechain: speech probability envelope (webrtcvad or silero via `ensure_model`)
  with 200ms lookahead → BGM gain curve (smooth, no pumping) instead of the current simple duck.
- **SFX layer:** ship a small original SFX kit (whooshes/impacts/risers generated procedurally
  like `audio/bgm.py` pads); auto-place on graphic hits, punch-ins, and B-roll cuts.

```
Tool: add_sfx(project_id, auto=True, events?: [{t, kind}], gain_db=-18) -> {n_placed}
```

### W5.4 Music that matches the edit
**Today:** procedural sine pads, one mood knob.
**Build (three tiers):**
1. **Procedural v2** (`audio/bgm.py` rewrite): chord progressions, tempo, intensity automation
   driven by the pacing/energy curve (W4.2 / W8.2) — music swells where the edit peaks;
   stems (pad/keys/pulse) rendered separately so ducking can keep the pulse and duck the pad.
2. **Beat grid:** librosa beat/tempo detection on any user-supplied BGM → expose `beat_grid`
   so cuts/B-roll/graphics snap to beat (`plan_cuts` and `insert_broll` gain `snap="beat"`).
3. **Generative hook:** `generate_music(prompt, duration, bpm?, sections?)` adapter interface —
   MusicGen (local, via `ensure_model`) or provider API (Stable Audio / others) behind
   `VIDMCP_MUSIC_PROVIDER`. Sections map to the pacing curve (intro/build/peak/outro).

### W5.5 Dubbing + voice cloning (multi-language versions)
**Build:** `vidmcp/audio/dubbing.py` — pipeline: transcript (have) → translate (host LLM via
MCP sampling, or provider) → TTS per segment with **duration fitting** (stretch ±15% to fit
original timing, re-flow captions otherwise) → new `dub_<lang>` track → mixdown variant.
- TTS adapters: macOS `say` (exists in `attach_narration`), **Coqui XTTS-v2** local cloning
  via `ensure_model` (consent-gated: requires explicit `voice_clone_consent=True` and logs
  provenance into the signed manifest — reuse `provenance/sign.py`), ElevenLabs/OpenAI TTS
  behind API keys.
- Optional lip-sync stage stubbed as an adapter (Wav2Lip-class models), off by default.

```
Tool: dub_video(project_id, language, voice="clone"|"neutral", voice_clone_consent=False) -> {track, caption_file, render?}
```

### W5.6 Loudness targeting per destination
**Today:** single `-14 LUFS` loudnorm.
**Build:** `vidmcp/audio/loudness.py` — target table: YouTube −14 LUFS / −1 dBTP,
Reels/TikTok −14/−1, podcast −16/−1.5, broadcast EBU R128 −23/−1. Two-pass loudnorm
(measure → apply, `measure_loudness` already exists). `export_render` / `export_multi`
pick the target from the preset automatically; report achieved LUFS/TP in the result.

### W5.7 Spatial polish
**Build:** subtle stereo widening on BGM/SFX only (Haas/mid-side, mono-compatible — check
correlation ≥ 0.5), voice stays center. De-esser and mouth-click reduction added to the voice
chain (`adeclick`, dynamic EQ around 6–9 kHz).

**Acceptance gates (W5):** mixdown hits target LUFS ±0.5 and TP ≤ ceiling on all presets;
stem-separated voice SNR beats filter chain on a noisy fixture; ducked BGM never masks speech
(speech-band BGM level ≥12 dB under voice during speech); dubbed segment durations within
±15% of originals; zero clicks at cut points (no sample discontinuity > −40 dBFS).

---

## W6. Motion Graphics, Brand, Style

### W6.1 MoGraph engine + declarative spec
**Today:** particles + flowfield (`effects/particles.py`), infographic cards
(`edit/infographics.py`), Manim/procedural scenes. No titles/lower-thirds/callouts system.
**Build:** `vidmcp/graphics/` — a keyframe animation engine (Pillow/Skia drawing, easing
library: cubic/spring/overshoot) with a declarative JSON spec:

```json
{"template": "lower_third", "t": 4.2, "duration": 3.5,
 "fields": {"title": "Dhiraj Lochib", "subtitle": "Time dilation, visually"},
 "anim": {"in": "slide_up_spring", "out": "fade"}, "brand": "auto"}
```
Ship templates: `title_card`, `lower_third`, `callout_arrow`, `stat_counter` (animated
number), `progress_bar`, `chapter_card`, `subscribe_reminder`, `quote_card`,
`animated_list`, `bar_chart` / `line_chart` (data-driven). All render as RGBA overlay layers
in the existing stack; all accept `z` for behind-subject placement (W1.3).
Port the HUD/chapter/glass-card visual language from `scripts/edit_dimensions_pro.py` — it is
already the brand look.

```
Tool: add_graphics(project_id, items: [spec]) -> {layer_ids, preview_pngs}
Tool: list_graphic_templates() -> templates + field schemas
```

### W6.2 Brand kit persistence
**Build:** `workspace/brand/<name>.json` (+ assets dir): fonts, palette, logo (+ placement +
watermark opacity), caption style defaults, LUT, safe margins, lower-third layout, SFX kit
choice, BGM style, outro card. Loaded once, auto-injected everywhere (captions
`captions/styles.py`, infographics, graphics templates, thumbnails, watermark on export).

```
Tool: set_brand_kit(name?, kit: dict) / get_brand_kit() / extract_brand_from_video(path)  # pulls dominant palette + font suggestion from an existing branded video
```

### W6.3 Speech-locked graphics v2
Wire graphics beats to the word timeline exactly as `add_speech_infographics` does, but for
every template — `add_graphics(items=[...], lock="speech:keyword")` places on keyword hit
with the existing speech-lock machinery (`audio/speech_lock.py`).

### W6.4 Style transfer + look emulation
**Build:** two tiers in `vidmcp/effects/style.py`:
1. **Look emulation (cheap, always works):** grade fingerprint (W2.3) + grain match (W1.3) +
   vignette/halation — "make it look like this reference" without neural cost.
2. **Neural style transfer layer:** ONNX fast-style-transfer models via `ensure_model`
   (a few shipped styles); temporal consistency via flow-warped previous-output blending
   (reuse W1.2 machinery). Applied `background_only` by default so faces are untouched.

### W6.5 Generative plates v2 + 2.5D parallax
**Today:** `enable_generative` flag exists, `effects/background.py` has a placeholder path;
Meshy plates exist (`integrations/meshy.py`).
**Build:** provider adapter `generate_plate(prompt, provider)` (image gen behind API keys) →
depth-estimate the still (Depth-Anything-V2, `ensure_model`) → **2.5D camera** (layered
parallax pan/zoom from the depth map) so generated backgrounds move like a filmed plate.
Combine with `reproject_background` so real camera motion drives the parallax.

### W6.6 Remotion: render, not scaffold
**Today:** `scaffold_remotion_scene` only writes a scaffold.
**Build:** if `node`+`npx` present: install a pinned Remotion project template into
workspace, inject props from graphics spec / brand kit, `npx remotion render` to a plate,
place as layer. Falls back to W6.1 native templates when node is absent.

**Acceptance gates (W6):** every template renders correctly at 16:9 and 9:16 (goldens);
brand kit swap changes fonts/colors on all surfaces with zero per-tool config; style transfer
temporal flicker (frame-diff in static regions) below threshold; graphics land within 50ms
of their locked words.

---

## W7. Recipe System v2 + Adaptive Templates

### W7.1 Recipe schema v2 (typed, composable)
**Today:** `harness/recipes.py` = static dicts with implicit magic keys; VidDSL exists
separately (`dsl/viddsl.py`).
**Build:** pydantic `RecipeV2` in `vidmcp/harness/recipe_schema.py`:
- `steps: [{tool, args, when?: expr, on_fail?: retry|skip|abort}]` — same expression language
  as `run_ops` conditions (one evaluator, `harness/expr.py`).
- `params:` typed, defaulted, surfaced to agents (`{aggressiveness: {type: float, default: .5, doc}}`).
- `extends: name` — inheritance with deep-merge; `compose: [a, b]` — sequential composition
  with declared merge policy for conflicting layers (last-wins | stack | error).
- `brand: auto` pulls the brand kit; `requires: [models/tools]` validated up front.
- v1 dicts auto-migrate (adapter keeps all 10 current recipes working).

```
Tool: compose_recipes(names: list[str], overrides?: dict) -> {recipe, conflicts_resolved}
Tool: validate_recipe(recipe: dict) -> {ok, errors, required_models}
```

### W7.2 Content-type detection → adaptive templates
**Build:** classifier in `vidmcp/harness/content_type.py` using the footage index (W4.2):
talking-head / tutorial-screencast / vlog-broll-heavy / product-demo / lecture / ad. Signals:
talking_head_score (exists), screen-content detection (flat regions + text density), shot
length distribution, speech ratio, scene variety.
- Each recipe may declare `adapt: {talking_head: {...param overrides}, tutorial: {...}}`.
- `run_intent` consults the classifier so *"edit this"* with no style words still picks a
  sane pipeline.

```
Tool: classify_content(project_id) -> {type, confidence, signals}
```

### W7.3 Recipe marketplace hardening
Signed recipes (reuse `provenance/sign.py` HMAC), semantic versioning, `requires` checked at
install (`marketplace/registry.py` extension). Recipe runs record recipe hash into the edit
graph for reproducibility.

**New shipped recipes (using everything above):** `podcast_clip_factory` (find best moments →
vertical clips w/ captions+punch-ins), `tutorial_supercut`, `ad_15s`, `cinematic_vlog`,
`lecture_to_shorts`, `multilang_release` (dub + export matrix), `music_video_beatcut`.

**Acceptance gates (W7):** all v1 recipes pass through the v2 engine unchanged (regression
fixtures); a composed recipe (`cinematic_bokeh` + `talking_head_tight`) produces both effects
with zero manual conflict fixes; classifier ≥80% on a labeled 20-clip fixture set.

---

## W8. Agent Cognition — Plan → Execute → Self-Critique → Repair

This is the workstream that makes it *agent-native* rather than a toolbox.

### W8.1 Persisted, executable edit plans
**Today:** `plan_edit` returns a throwaway list; `propose_edit_strategies` debates but output
isn't executable.
**Build:** `vidmcp/agents/edit_plan.py`:
- `EditPlan` = persisted artifact on the edit graph (`advanced/causal_graph.py`): ordered
  steps (tool+args+expected outcome+cost estimate), narrative summary, pacing spec (W8.2),
  alternates considered. Content-addressed, branchable, diffable — reuse graph branches for
  plan variants.
- `draft_edit_plan` composes: footage index (W4) + content type (W7.2) + intent + recipe
  candidates → a full plan **without side effects**, returning a compact human/agent-readable
  brief. The host LLM can rewrite any step (`revise_edit_plan(plan_id, patch)`).
- `execute_edit_plan(plan_id)` runs via `run_ops` (W10.3) with checkpointing: each step
  commits to the edit graph; failure → the plan's `on_fail` policy → gate-driven repair (below)
  → resume from checkpoint, never from scratch.

```
Tool: draft_edit_plan(project_id, intent, duration_target?, deliverables?) -> {plan_id, brief, est_cost_s, steps}
Tool: revise_edit_plan(plan_id, patch) -> {plan_id', diff}
Tool: execute_edit_plan(plan_id, until_step?) -> {checkpoints, results, gate_report}
```

### W8.2 Creative intent model — pacing, tone, arc
**Build:** `vidmcp/agents/creative.py` — makes "feel" measurable so agents can reason about it:
- **Pacing curve:** target cut-density/energy over time (templates: `hook_heavy_short`,
  `steady_educational`, `build_to_peak`). Compare measured curve (from cut plan + motion +
  music intensity) against target; deviations become actionable notes ("2:10–2:40 sags:
  tighten cuts or add punch-in/B-roll").
- **Tone profile:** {energetic, calm, dramatic, playful, premium} → concrete parameter
  mappings (grade choice, BGM style, SFX density, caption animation, punch-in frequency).
  Recipes and `draft_edit_plan` consume tone instead of hardcoding params.

```
Tool: analyze_pacing(project_id) -> {curve, cut_density, sag_regions, hook_strength_first_5s}
Tool: set_creative_profile(project_id, tone, pacing_template) -> applied defaults
```

### W8.3 Self-critique v2 — actually re-watch the render
**Today:** critic = numeric matte/coverage metrics (`agents/critic.py`, `critics/ensemble.py`).
**Build:** `vidmcp/agents/rewatch.py` — post-render QC pass:
- **Mechanical audio QC:** A/V sync drift (cross-correlate source vs render speech envelope),
  clipping, LUFS vs target, dead-air stretches, click detection at every cut point.
- **Mechanical visual QC:** black/frozen frames, matte edge flicker on the *render* (not just
  masks), caption/graphic overflow outside safe margins, caption-word mismatch vs transcript.
- **Perceptual QC via host LLM (MCP sampling):** sample frame strips + scopes + captions at
  cut points and graphic beats → structured critique request ("rate cut smoothness, matte
  believability, graphic legibility; list defects with timestamps"). Uses MCP sampling so it
  costs the *host's* model, not a bundled one; degrade gracefully to mechanical-only.
- Output = defect list with severities + **repair routes** (mapping defect→tool call, like
  `fix_route` in the critic ensemble today). `execute_edit_plan` auto-runs up to
  `settings.max_repair_passes` repairs.

```
Tool: rewatch_render(project_id, depth="mechanical"|"full") -> {defects[{t, kind, severity, repair}], scores}
```

### W8.4 Token/call efficiency (extend what 2250d81 started)
- All new tools return compact results through `utils/compact.py`; heavy artifacts by path.
- `search_footage`/`suggest_broll`/`plan_cuts` return top-k with reasons, never raw arrays.
- New packs in `harness/packs.py`: `editor` (understanding+cutting+delivery),
  `colorist` (W2), `sound` (W5), `mograph` (W6); the god-path pack `director` = draft/execute
  plan + rewatch + delivery only (~12 tools) — an agent can run the entire north-star scenario
  inside `director` without ever seeing the other ~120 schemas.
- Failure results always carry `fix_hint` + the exact tool call to try next (pattern already
  in gates; enforce via a result contract in `harness/contracts.py`).

**Acceptance gates (W8):** north-star scenario runs end-to-end via
`draft_edit_plan → execute_edit_plan → rewatch_render` in ≤6 MCP calls from the host agent;
a seeded defect (forced matte flicker + LUFS miss) is caught by rewatch and auto-repaired
within 2 repair passes; plan checkpoint resume verified by killing execution mid-plan.

---

## W9. Delivery & Distribution

### W9.1 Batch multi-platform export
**Today:** one preset per `export_render` call; scale/pad only; CRF 18 everywhere.
**Build:** `vidmcp/media/delivery.py`:
- Target spec table: resolution, fps policy, codec (h264 high / hevc / av1 when encoder
  present), bitrate ladder or CRF, pixel format, loudness target (W5.6), container, max
  duration hints (Shorts ≤60s etc.).
- `export_multi` renders the mezzanine **once**, then derives all targets in parallel
  (reframe track from W3.1 applied per aspect; captions re-laid-out per aspect — 9:16 gets
  bigger, higher-placed captions automatically).

```
Tool: export_multi(project_id, targets=["youtube_16x9","reels_9x16","square_1x1"], hw=True)
      -> [{target, path, codec, bitrate, lufs, true_peak, size_mb}]
```

### W9.2 Thumbnail A/B generation
**Today:** `generate_thumbnail` = mid-frame + title.
**Build:** `vidmcp/media/thumbs.py`:
- Candidate frames scored by: face size × expression intensity (W4.2) × sharpness × aesthetic
  score × rule-of-thirds subject placement; take top-N distinct frames.
- Compose variants: brand-kit title styles, optional matte cutout with contrasting BG
  (subject pop-out — reuse W1 alpha), color pop variant.
- Emit N variants + a contact sheet; host agent picks or A/B tests.

```
Tool: generate_thumbnails(project_id, n=3, title_variants?: list[str]) -> {variants[...], contact_sheet}
```

### W9.3 Metadata pack
**Build:** from transcript + scenes: chapters (`00:00 Hook…` from scene/topic boundaries),
title candidates, description, hashtags, SRT + VTT sidecars (already have ASS internals —
add converters in `captions/burn.py`). One call, publish-ready.

```
Tool: generate_metadata(project_id, platform="youtube") -> {chapters, titles[], description, tags[], srt, vtt}
```

### W9.4 Clip factory (long → shorts)
**Build:** compose W4 search + emotion curve + W8 pacing: rank self-contained high-energy
segments (hook strength, quote-ability, complete-thought boundary detection from transcript),
then run the vertical pipeline per clip.

```
Tool: extract_clips(project_id, n=3, max_sec=45) -> [{t_start, t_end, hook_score, reason}]
```
(This + `export_multi` + `generate_thumbnails` = the `podcast_clip_factory` recipe from W7.)

**Acceptance gates (W9):** `export_multi` produces 3 targets ≤1.6× the wall time of the
slowest single target; every target passes its own loudness gate; thumbnails contain a
detected face of ≥18% frame height on talking-head fixtures; chapters align with scene
boundaries ±2s.

---

## 3. Consolidated New Tool Surface

~40 new/upgraded tools. Pack assignment keeps default surfaces small.

| Tool | Workstream | Packs |
|---|---|---|
| `ensure_model`, `list_models` | W10 | admin, always |
| `run_ops` | W10 | always |
| `refine_alpha`, `stabilize_matte` | W1 | vfx |
| `detect_scenes`, `build_footage_index`, `search_footage`, `search_library` | W4 | editor, education |
| `plan_cuts`, `apply_cut_plan` | W4 | editor, talking_head |
| `suggest_broll`, `insert_broll` | W4 | editor |
| `apply_lut`, `list_luts`, `auto_color`, `match_color`, `color_scopes` | W2 | colorist, vfx |
| `smart_reframe`, `add_camera_moves`, `time_warp`, `stabilize_video` | W3 | editor |
| `audio_tracks`, `edit_audio_track`, `mixdown_audio` | W5 | sound |
| `add_sfx`, `generate_music`, `dub_video` | W5 | sound |
| `add_graphics`, `list_graphic_templates` | W6 | mograph, talking_head |
| `set_brand_kit`, `get_brand_kit`, `extract_brand_from_video` | W6 | always |
| `generate_plate` (v2), style transfer layer | W6 | vfx, mograph |
| `compose_recipes`, `validate_recipe`, `classify_content` | W7 | always |
| `draft_edit_plan`, `revise_edit_plan`, `execute_edit_plan` | W8 | director, always |
| `analyze_pacing`, `set_creative_profile`, `rewatch_render` | W8 | director, editor |
| `export_multi`, `generate_thumbnails`, `generate_metadata`, `extract_clips` | W9 | director, editor, talking_head |

New packs: `editor`, `colorist`, `sound`, `mograph`, `director`. `director` ≈ 12 tools and can
execute the whole north-star scenario.

---

## 4. Dependency / Model Matrix (all optional extras, all via `ensure_model`)

| Capability | Primary | Fallback (always works) |
|---|---|---|
| Video matting | MatAnyone / RVM / ViTMatte | guided-filter alpha snap |
| Shot detection | TransNetV2 | PySceneDetect → histogram |
| Diarization | pyannote 3.1 | spectral clustering (exists) |
| ASR | faster-whisper large-v3-turbo | base (exists) → energy align (exists) |
| Audio events | PANNs / AST | energy heuristics |
| Embeddings | SigLIP / CLIP + MiniLM | histogram + keyword match |
| Stems | Demucs htdemucs | afftdn chain (exists) |
| VAD | silero-vad | webrtcvad → energy |
| Depth | Depth-Anything-V2 | multi-cue (exists) |
| Frame interp | RIFE | ffmpeg minterpolate → blend |
| TTS/cloning | XTTS-v2 / provider APIs | macOS say / espeak (exists) |
| Music gen | MusicGen / provider APIs | procedural v2 |
| Style transfer | ONNX fast-style | grade-fingerprint look emulation |
| Saliency | U²-Net | matte+face+motion fusion |

pyproject extras: `understand` (transnet, siglip, minilm, pann), `matte-pro`, `sound-pro`
(demucs, silero, rubberband), `dub` (xtts), `pro = all of the above`. Core install stays lean;
**every feature has a dependency-free fallback path** — this is already the codebase's
strongest design principle (mock SAM, procedural scenes, energy diarization). Preserve it.

---

## 5. Quality Gate Additions (config.py)

```
min_alpha_quality, max_edge_flicker_dtssd          (W1)
max_color_cast_deltaE, max_skin_deltaE, max_clip_pct (W2)
max_crop_jerk, reframe_face_containment            (W3)
lufs_tolerance, max_true_peak, min_speech_bgm_separation_db, max_av_sync_ms (W5)
caption_safe_margin_violations = 0                 (W6)
min_hook_score_short_form                          (W9)
max_repair_passes = 2                              (W8)
```
All surfaced through `evaluate_quality_gates` with `fix_hint` + exact repair tool call.

## 6. Risks & Constraints

- **Latency:** understanding pass + alpha refinement are the heavy new costs → they are why
  W10 (proxy + cache + async) is first. Budget: full north-star scenario ≤ 3× realtime on
  M-series at proxy quality, final render excluded.
- **Voice cloning:** consent flag mandatory, provenance-signed, off by default. Never clone
  without `voice_clone_consent=True`.
- **Model licenses:** registry records license per model; `pro` extras exclude non-commercial
  weights by default (pyannote/XTTS need review; document in registry).
- **Backward compat:** every existing tool keeps working; v1 recipes auto-migrate; new
  behavior is opt-in flags or new tools.
- **Scope discipline for the implementing agent:** land each workstream's acceptance gates
  green (add fixtures under `tests/`) before starting the next; the order in §2 is the
  execution order.
