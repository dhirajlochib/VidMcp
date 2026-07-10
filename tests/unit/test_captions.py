"""Caption cues + ASS."""

from __future__ import annotations

from pathlib import Path

from vidmcp.captions.burn import words_to_cues, write_ass


def test_words_to_cues_and_ass(tmp_path: Path):
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.3},
        {"word": "world", "start": 0.3, "end": 0.6},
        {"word": "this", "start": 0.6, "end": 0.8},
        {"word": "is", "start": 0.8, "end": 1.0},
        {"word": "a", "start": 1.0, "end": 1.1},
        {"word": "test", "start": 1.1, "end": 1.4},
    ]
    cues = words_to_cues(words, max_chars=20)
    assert cues
    ass = tmp_path / "c.ass"
    write_ass(cues, ass, style="brand")
    text = ass.read_text()
    assert "Dialogue:" in text
    assert "Hello" in text or "world" in text
