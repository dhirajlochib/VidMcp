from vidmcp.agents.planner import PlannerAgent


def test_cyberpunk_plan():
    plan = PlannerAgent().plan(
        "Turn this talking-head video into a cyberpunk style with dramatic particle effects behind the speaker"
    )
    tools = [s.tool for s in plan.steps]
    assert "segment_subject" in tools
    assert "apply_background_effects" in tools
    assert "composite_and_render" in tools
    assert "review_edit" in tools
    assert "cyberpunk" in plan.style_tags
    effects = PlannerAgent().effects_from_tags(plan.style_tags, plan.intent)
    types = {e["effect_type"] for e in effects}
    assert "cyberpunk" in types
    assert "particles" in types
