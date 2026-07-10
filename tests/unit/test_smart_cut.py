"""Smart cut planner."""

from vidmcp.edit.smart_cut import plan_smart_cuts


def test_plan_removes_gap_and_filler():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "basically", "start": 0.5, "end": 1.2},
        {"word": "world", "start": 3.5, "end": 4.0},
    ]
    ranges = plan_smart_cuts(words, duration_sec=5.0, min_gap=0.45, aggressiveness=0.7)
    kept = sum(r.end - r.start for r in ranges)
    assert kept < 5.0
    # gap ~2.3s should be largely removed
    assert kept < 4.0
