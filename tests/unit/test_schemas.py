from vidmcp.models.schemas import AnalyzeVideoResult, SegmentSubjectResult


def test_tool_response_defaults():
    a = AnalyzeVideoResult(ok=True, project_id="x", message="ok")
    assert a.suggested_prompts == []
    s = SegmentSubjectResult(ok=True, prompt="person", backend="mock")
    assert s.object_count == 0
