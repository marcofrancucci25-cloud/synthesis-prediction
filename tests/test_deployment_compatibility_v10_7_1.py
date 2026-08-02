from pathlib import Path


def test_optional_evidence_api_is_not_accessed_directly_at_import_time():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "verified_precedents = engine.verified_precedents" not in source
    assert 'getattr(engine, "verified_precedents", None)' in source


def test_release_still_includes_the_10_7_1_deployment_fix():
    version = tuple(int(part) for part in Path("VERSION").read_text(encoding="utf-8").strip().split("."))
    assert version >= (10, 7, 1)
