from src.reports.template import build_report_payload


def test_report_contains_limitations() -> None:
    report = build_report_payload("broken", 0.9, [])
    assert report["requiere_revision_humana"] is True
    assert "no constituye diagnóstico" in report["limitacion"]
