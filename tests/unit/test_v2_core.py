"""V2 upgrade core tests — expr, LUT, cuts, matte metrics, audio, recipes, graphics."""

from __future__ import annotations

import numpy as np
import pytest

from vidmcp.harness.expr import evaluate


class TestExpr:
    def test_comparison_and_dotted_lookup(self):
        ctx = {"seg": {"temporal_stability": 0.5, "ok": True}}
        assert evaluate("seg.temporal_stability < 0.65", ctx) is True
        assert evaluate("seg.temporal_stability > 0.65", ctx) is False
        assert evaluate("seg.ok and seg.temporal_stability < 1", ctx) is True

    def test_missing_names_are_none_safe(self):
        assert evaluate("nope.thing > 3", {}) is False
        assert evaluate("nope.thing == None", {}) is True

    def test_no_calls_allowed(self):
        with pytest.raises(Exception):
            evaluate("__import__('os').system('x')", {})

    def test_empty_is_true(self):
        assert evaluate("", {}) is True


class TestLut:
    def test_identity_builtin_close(self):
        from vidmcp.color.lut import apply_lut, builtin_table

        img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        # filmic_soft at intensity 0 == identity
        out = apply_lut(img, builtin_table("filmic_soft"), intensity=0.0)
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1

    def test_cube_roundtrip(self, tmp_path):
        from vidmcp.color.lut import apply_lut, builtin_table, parse_cube, write_cube

        table = builtin_table("teal_orange", size=17)
        p = write_cube(table, tmp_path / "t.cube")
        loaded = parse_cube(p)
        img = np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        a = apply_lut(img, table)
        b = apply_lut(img, loaded)
        assert np.abs(a.astype(int) - b.astype(int)).max() <= 2

    def test_unknown_look_raises(self):
        from vidmcp.color.lut import resolve_table

        with pytest.raises((KeyError, FileNotFoundError)):
            resolve_table("not_a_look")


class TestCutPlanner:
    def test_pause_classification(self):
        from vidmcp.edit.cut_planner import classify_pause

        assert classify_pause(0.2, "and", False, 0.3, 0.5) == "breath"
        assert classify_pause(1.2, "nothing", True, 0.8, 0.5) == "dramatic"
        assert classify_pause(1.5, "and", False, 0.2, 0.5) == "dead"

    def test_contextual_filler(self):
        from vidmcp.edit.cut_planner import _is_contextual_filler

        assert _is_contextual_filler("um", "", 0.1, 0.1) is True
        # 'like' as comparator survives
        assert _is_contextual_filler("like", "feels", 0.05, 0.05) is False
        # isolated 'like' dies
        assert _is_contextual_filler("like", "and", 0.6, 0.4) is True

    def test_retake_detection(self):
        from vidmcp.edit.cut_planner import detect_retakes

        sents = [
            {"text": "so today we are going to learn about time dilation", "start": 0.0, "end": 3.0},
            {"text": "so today we are going to learn about time dilation okay", "start": 3.5, "end": 6.5},
            {"text": "completely different sentence here", "start": 7.0, "end": 9.0},
        ]
        removals = detect_retakes(sents)
        assert len(removals) == 1
        assert removals[0][0] == 0.0  # earlier take removed


class TestMatteMetrics:
    def test_dtssd_flat_is_zero(self):
        from vidmcp.matte.temporal import dtssd

        masks = [np.full((32, 32), 128, np.uint8)] * 5
        assert dtssd(masks) == 0.0

    def test_dtssd_flicker_positive(self):
        from vidmcp.matte.temporal import dtssd

        a = np.zeros((32, 32), np.uint8)
        b = np.full((32, 32), 255, np.uint8)
        assert dtssd([a, b, a, b]) > 0.5

    def test_trimap_bands(self):
        from vidmcp.matte.alpha_refine import trimap_from_mask

        mask = np.zeros((64, 64), np.uint8)
        mask[16:48, 16:48] = 255
        fg, bg, unknown = trimap_from_mask(mask, band_px=8)
        assert fg.sum() > 0 and bg.sum() > 0 and unknown.sum() > 0
        assert not (fg & bg).any()

    def test_guided_alpha_range(self):
        from vidmcp.matte.alpha_refine import refine_frame_alpha

        frame = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[20:44, 20:44] = 255
        alpha = refine_frame_alpha(frame, mask)
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0
        assert alpha[30, 30] == 1.0  # definite FG stays opaque


class TestAudio:
    def test_loudness_targets(self):
        from vidmcp.audio.loudness import target_for

        assert target_for("youtube")["lufs"] == -14.0
        assert target_for("podcast_audio")["lufs"] == -16.0
        assert target_for("broadcast")["lufs"] == -23.0

    def test_ducking_reduces_bgm_under_speech(self):
        from vidmcp.audio.tracks import SR, duck_music

        n = SR * 2
        voice = np.zeros((n, 2), np.float32)
        voice[SR // 2 : SR + SR // 2] = 0.4  # speech in the middle second
        bgm = np.full((n, 2), 0.3, np.float32)
        ducked = duck_music(bgm, voice)
        mid = np.abs(ducked[SR - 1000 : SR + 1000]).mean()
        edge = np.abs(ducked[: SR // 8]).mean()
        assert mid < edge * 0.7

    def test_sfx_generators_shapes(self):
        from vidmcp.audio.sfx import impact, riser, whoosh

        for gen in (whoosh, impact, riser):
            clip = gen()
            assert clip.ndim == 2 and clip.shape[1] == 2
            assert np.abs(clip).max() <= 4.0

    def test_music_synthesis_stems(self):
        from vidmcp.audio.music import synthesize_score

        stems = synthesize_score(2.0, style="uplifting", bpm=100, intensity_curve=[0.2, 0.9])
        assert set(stems) == {"pad", "keys", "pulse"}
        for buf in stems.values():
            assert buf.shape[1] == 2 and len(buf) > 0


class TestRecipesV2:
    def test_resolve_and_validate_shipped(self):
        from vidmcp.harness.recipe_schema import RECIPES_V2, validate_recipe

        for name in RECIPES_V2:
            v = validate_recipe({"name": "inline", "steps": []}) if False else validate_recipe(
                RECIPES_V2[name].model_dump()
            )
            assert v["ok"], f"{name}: {v['errors']}"

    def test_compose_dedupes_renders(self):
        from vidmcp.harness.recipe_schema import compose_recipes

        out = compose_recipes(["cinematic_vlog", "ad_15s"])
        assert out["ok"]
        tools = [s["tool"] for s in out["recipe"]["steps"]]
        renders = [t for t in tools if t in ("composite_and_render", "export_render", "export_multi")]
        assert len(renders) == 1  # last recipe's delivery wins

    def test_unknown_recipe(self):
        from vidmcp.harness.recipe_schema import compose_recipes

        assert compose_recipes(["nope_recipe"])["ok"] is False


class TestOpsTemplating:
    def test_dollar_reference_resolution(self):
        from vidmcp.harness.ops import _resolve_args

        results = {"cuts": {"plan_id": "cuts_abc123", "removed_sec": 4.2}}
        args = _resolve_args({"plan_id": "$cuts.plan_id", "n": 3, "nested": {"x": "$cuts.removed_sec"}}, results)
        assert args["plan_id"] == "cuts_abc123"
        assert args["nested"]["x"] == 4.2
        assert args["n"] == 3

    def test_op_table_targets_resolve(self):
        from vidmcp.harness.ops import OP_TABLE, resolve_op

        for name in OP_TABLE:
            fn = resolve_op(name)
            assert callable(fn), name


class TestContentType:
    def test_talking_head_signals(self):
        from vidmcp.harness.content_type import classify_signals

        t, conf = classify_signals({
            "talking_head_score": 0.9, "face_presence": 0.95, "speech_ratio": 0.8,
            "shots_per_min": 1.0, "motion": 0.1, "duration_sec": 300, "visual_flatness": 0.2,
        })
        assert t == "talking_head" and conf > 0.5

    def test_ad_signals(self):
        from vidmcp.harness.content_type import classify_signals

        t, _ = classify_signals({
            "talking_head_score": 0.1, "face_presence": 0.2, "speech_ratio": 0.3,
            "shots_per_min": 20.0, "motion": 0.8, "duration_sec": 30, "visual_flatness": 0.1,
        })
        assert t == "ad"


class TestGraphics:
    def test_all_templates_render_rgba(self):
        from vidmcp.graphics.brand import DEFAULT_KIT
        from vidmcp.graphics.templates import TEMPLATES

        fields = {
            "title": "Test", "subtitle": "Sub", "text": "Hello world", "author": "A",
            "value": 42, "label": "views", "number": "01", "items": ["one", "two"],
            "data": [{"label": "A", "value": 3}], "values": [1, 2, 3], "fraction": 0.5,
        }
        for name, fn in TEMPLATES.items():
            for p in (0.05, 0.5, 0.95):
                img = fn(640, 360, p, fields, DEFAULT_KIT)
                assert img.mode == "RGBA", name
                assert img.size == (640, 360), name

    def test_brand_color_parse(self):
        from vidmcp.graphics.brand import DEFAULT_KIT, color

        assert color(DEFAULT_KIT, "primary") == (0xD4, 0xFF, 0x2A)


class TestCreative:
    def test_tone_profiles_complete(self):
        from vidmcp.agents.creative import PACING_TEMPLATES, TONE_PROFILES

        for t, p in TONE_PROFILES.items():
            assert "lut" in p and "bgm_style" in p, t
        for name, fn in PACING_TEMPLATES.items():
            v = float(fn(0.5))
            assert 0.0 <= v <= 1.2, name


class TestFrameCache:
    def test_key_stability_and_cache(self, tmp_path):
        from vidmcp.core.framecache import OpCache, op_key

        f = tmp_path / "in.txt"
        f.write_text("hello")
        k1 = op_key("grade", {"lut": "noir"}, [f])
        k2 = op_key("grade", {"lut": "noir"}, [f])
        assert k1 == k2
        assert op_key("grade", {"lut": "teal_orange"}, [f]) != k1
        cache = OpCache(tmp_path)
        assert cache.get(k1) is None
        cache.put(k1, {"ok": True, "layer_id": "x"})
        hit = cache.get(k1)
        assert hit["ok"] and hit["cache_hit"]


class TestMetadata:
    def test_srt_cue_grouping(self):
        from vidmcp.media.metadata import _words_to_srt_cues

        words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(30)]
        cues = _words_to_srt_cues(words)
        assert len(cues) >= 2
        assert all(c["end"] > c["start"] for c in cues)

    def test_timestamp_formats(self):
        from vidmcp.media.metadata import _ts_ch, _ts_srt

        assert _ts_ch(75) == "1:15"
        assert _ts_ch(3700) == "1:01:40"
        assert _ts_srt(1.5) == "00:00:01,500"


class TestModelsRegistry:
    def test_registry_entries_wellformed(self):
        from vidmcp.models_registry import REGISTRY, ensure_model

        for name, entry in REGISTRY.items():
            assert entry.get("purpose") and entry.get("license"), name
        st = ensure_model("rvm")
        assert "found" in st and st["ok"]

    def test_unknown_model(self):
        from vidmcp.models_registry import ensure_model

        assert ensure_model("nope")["ok"] is False
