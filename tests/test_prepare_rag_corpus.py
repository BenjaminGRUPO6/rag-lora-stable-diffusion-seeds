from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_rag_corpus import METADATA_COLUMNS, process_corpus


def test_process_corpus_accepts_text_and_rejects_empty_and_duplicates(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    accepted = tmp_path / "accepted"
    rejected = tmp_path / "rejected"
    metadata = tmp_path / "metadata" / "document_sources.csv"
    results = tmp_path / "results"
    inbox.mkdir()

    corpus_text = (
        "Soybean seed quality standard for intact sound seed and germination. "
        "Broken split cracked seed indicates mechanical damage during handling, "
        "harvest, threshing, processing, transport and conditioning. "
        "Immature green seed maturity maturation and desiccation are relevant. "
        "Spotted visible surface alterations include stain, discoloration and mottled appearance. "
        "Seed coat skin damage, testa, hull and wrinkled seed coat are noted. "
        "Storage moisture humidity drying temperature and stored viability loss are covered."
    )
    (inbox / "soybean_quality.md").write_text(corpus_text, encoding="utf-8")
    (inbox / "duplicate.txt").write_text(corpus_text, encoding="utf-8")
    (inbox / "empty.txt").write_text("   \n\t", encoding="utf-8")

    records, summary = process_corpus(
        input_dir=inbox,
        accepted_dir=accepted,
        rejected_dir=rejected,
        metadata_path=metadata,
        results_dir=results,
        base_dir=tmp_path,
    )

    assert summary["documents_found"] == 3
    assert summary["accepted"] == 1
    assert summary["rejected"] == 2
    assert summary["duplicates"] == 1
    assert all(count >= 1 for count in summary["coverage_by_topic"].values())
    assert not summary["missing_topics"]
    assert (accepted / "duplicate.txt").exists()
    assert (rejected / "empty.txt").exists()
    assert (rejected / "soybean_quality.md").exists()
    assert (inbox / "soybean_quality.md").read_text(encoding="utf-8") == corpus_text

    accepted_records = [record for record in records if record.status == "accepted"]
    assert len(accepted_records) == 1
    assert accepted_records[0].authors == ""
    assert accepted_records[0].year == ""
    assert accepted_records[0].source_url == ""

    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == METADATA_COLUMNS
        rows = list(reader)
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"accepted", "rejected"}

    rejected_rows = list(csv.DictReader((results / "rejected_documents.csv").open(encoding="utf-8")))
    assert {row["notes"] for row in rejected_rows} == {
        "exact duplicate of duplicate.txt",
        "empty extracted text",
    }

    report = (results / "corpus_report.md").read_text(encoding="utf-8")
    assert "spotted/visible surface alterations" in report
    assert "fungus" not in report.lower()


def test_process_corpus_reports_unsupported_files(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    accepted = tmp_path / "accepted"
    rejected = tmp_path / "rejected"
    metadata = tmp_path / "metadata.csv"
    results = tmp_path / "results"
    inbox.mkdir()
    (inbox / "notes.docx").write_bytes(b"not a supported document")

    _, summary = process_corpus(
        input_dir=inbox,
        accepted_dir=accepted,
        rejected_dir=rejected,
        metadata_path=metadata,
        results_dir=results,
        base_dir=tmp_path,
    )

    assert summary["documents_found"] == 1
    assert summary["accepted"] == 0
    assert summary["rejected"] == 1
    assert summary["missing_topics"] == list(summary["coverage_by_topic"].keys())
    assert (rejected / "notes.docx").exists()

    summary_json = json.loads((results / "corpus_summary.json").read_text(encoding="utf-8"))
    assert summary_json["rejected"] == 1
