from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
METADATA_COLUMNS = [
    "document_id",
    "title",
    "authors",
    "year",
    "organization",
    "source_url",
    "license",
    "local_path",
    "sha256",
    "file_type",
    "pages",
    "language",
    "topics",
    "status",
    "notes",
]
REQUIRED_TOPICS = [
    "intact/general quality",
    "broken/mechanical damage",
    "immature",
    "spotted/visible surface alterations",
    "skin_damaged",
    "storage",
    "handling",
]
TOPIC_KEYWORDS = {
    "intact/general quality": [
        "general quality",
        "quality",
        "grade",
        "grading",
        "standard",
        "sound seed",
        "seed lot",
        "germination",
        "viability",
        "intact",
        "pure seed",
    ],
    "broken/mechanical damage": [
        "broken",
        "split",
        "splits",
        "cracked",
        "crack",
        "mechanical damage",
        "physical damage",
        "fracture",
        "breakage",
        "damage",
    ],
    "immature": [
        "immature",
        "green seed",
        "green seeds",
        "maturity",
        "maturation",
        "desiccation",
        "ripening",
        "physiological maturity",
    ],
    "spotted/visible surface alterations": [
        "spotted",
        "spot",
        "spots",
        "stain",
        "staining",
        "purple seed stain",
        "discoloration",
        "discoloured",
        "discolored",
        "mottled",
        "visible",
        "surface alteration",
        "surface",
        "appearance",
    ],
    "skin_damaged": [
        "skin damage",
        "seed coat",
        "seedcoat",
        "coat damage",
        "seed coat damage",
        "testa",
        "hull",
        "wrinkled",
        "wrinkle",
    ],
    "storage": [
        "storage",
        "stored",
        "moisture",
        "humidity",
        "drying",
        "temperature",
        "warehouse",
        "viability loss",
    ],
    "handling": [
        "handling",
        "harvest",
        "harvesting",
        "combine",
        "threshing",
        "cleaning",
        "processing",
        "transport",
        "conditioning",
    ],
}


@dataclass(frozen=True)
class ExtractedDocument:
    """Text and basic technical metadata extracted from a local document."""

    text: str
    pages: int | None
    title: str


@dataclass(frozen=True)
class CorpusRecord:
    """Normalized metadata record for one processed corpus file."""

    document_id: str
    title: str
    authors: str
    year: str
    organization: str
    source_url: str
    license: str
    local_path: str
    sha256: str
    file_type: str
    pages: str
    language: str
    topics: str
    status: str
    notes: str

    def as_row(self) -> dict[str, str]:
        """Return this record in CSV column order."""
        return {column: getattr(self, column) for column in METADATA_COLUMNS}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for corpus preparation."""
    parser = argparse.ArgumentParser(description="Prepare and validate the RAG document corpus.")
    parser.add_argument("--input", type=Path, default=Path("data/documents/inbox"))
    parser.add_argument("--accepted", type=Path, default=Path("data/documents/accepted"))
    parser.add_argument("--rejected", type=Path, default=Path("data/documents/rejected"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/document_sources.csv"))
    parser.add_argument("--results", type=Path, default=Path("results/rag"))
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_document(path: Path) -> ExtractedDocument:
    """Extract plain text and page count from a supported document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - exercised only without the optional package
            raise RuntimeError(
                "No se puede extraer PDF porque falta pypdf. "
                "Instala requirements-rag.txt para procesar documentos PDF."
            ) from exc
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        metadata_title = ""
        if reader.metadata and reader.metadata.title:
            metadata_title = str(reader.metadata.title).strip()
        return ExtractedDocument(text=text, pages=len(reader.pages), title=metadata_title)

    if suffix in {".txt", ".md", ".markdown"}:
        return ExtractedDocument(text=path.read_text(encoding="utf-8-sig"), pages=None, title="")

    raise ValueError(f"Formato no soportado: {suffix}")


def detect_language(text: str) -> str:
    """Detect English or Spanish with a deterministic local heuristic."""
    lowered = f" {text.lower()} "
    spanish_hits = sum(
        lowered.count(token)
        for token in [
            " el ",
            " la ",
            " de ",
            " que ",
            " para ",
            " semilla",
            " semillas",
            " soja",
            " calidad",
            " almacenamiento",
            " manejo",
        ]
    )
    english_hits = sum(
        lowered.count(token)
        for token in [
            " the ",
            " and ",
            " of ",
            " for ",
            " seed",
            " seeds",
            " soybean",
            " quality",
            " storage",
            " handling",
        ]
    )
    if spanish_hits == 0 and english_hits == 0:
        return "unknown"
    return "es" if spanish_hits > english_hits else "en"


def assign_topics(text: str) -> list[str]:
    """Assign corpus topics based on visible quality and handling terminology."""
    lowered = text.lower()
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return topics


def normalize_relative(path: Path, base_dir: Path) -> str:
    """Return a POSIX-style relative path when possible."""
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_unique_destination(source: Path, destination_dir: Path) -> Path:
    """Choose a destination path without overwriting an earlier copied file."""
    destination = destination_dir / source.name
    if not destination.exists() or compute_sha256(destination) == compute_sha256(source):
        return destination
    stem = destination.stem
    suffix = destination.suffix
    counter = 2
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_document(source: Path, destination_dir: Path) -> Path:
    """Copy a document to a target directory while preserving the original."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = ensure_unique_destination(source, destination_dir)
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def load_existing_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Load existing metadata keyed by local path and checksum when available."""
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return {}

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    existing: dict[str, dict[str, str]] = {}
    for row in rows:
        normalized = {
            "document_id": row.get("document_id", ""),
            "title": row.get("title", ""),
            "authors": row.get("authors", row.get("author_or_institution", "")),
            "year": row.get("year", ""),
            "organization": row.get("organization", row.get("publisher", "")),
            "source_url": row.get("source_url", row.get("url_or_doi", "")),
            "license": row.get("license", row.get("license_or_access", "")),
            "local_path": row.get("local_path", row.get("file_path", "")),
            "sha256": row.get("sha256", ""),
        }
        for key in (normalized["local_path"], normalized["sha256"]):
            if key:
                existing[key] = normalized
    return existing


def next_document_id(existing_ids: Iterable[str], index: int) -> str:
    """Create a stable sequential document identifier for new rows."""
    used_numbers = []
    for document_id in existing_ids:
        if document_id.startswith("DOC") and document_id[3:].isdigit():
            used_numbers.append(int(document_id[3:]))
    base = max(used_numbers, default=0)
    return f"DOC{base + index:03d}"


def build_record(
    *,
    source: Path,
    copied_path: Path,
    base_dir: Path,
    sha256: str,
    extracted: ExtractedDocument | None,
    status: str,
    notes: list[str],
    existing: dict[str, str],
    document_id: str,
) -> CorpusRecord:
    """Build one normalized metadata row without inventing bibliographic fields."""
    text = extracted.text if extracted else ""
    topics = assign_topics(text) if text else []
    language = detect_language(text) if text else "unknown"
    return CorpusRecord(
        document_id=existing.get("document_id") or document_id,
        title=existing.get("title") or (extracted.title if extracted else ""),
        authors=existing.get("authors", ""),
        year=existing.get("year", ""),
        organization=existing.get("organization", ""),
        source_url=existing.get("source_url", ""),
        license=existing.get("license", ""),
        local_path=normalize_relative(copied_path, base_dir),
        sha256=sha256,
        file_type=source.suffix.lower().lstrip("."),
        pages=str(extracted.pages) if extracted and extracted.pages is not None else "",
        language=language,
        topics="; ".join(topics),
        status=status,
        notes="; ".join(notes),
    )


def process_corpus(
    *,
    input_dir: Path,
    accepted_dir: Path,
    rejected_dir: Path,
    metadata_path: Path,
    results_dir: Path,
    base_dir: Path | None = None,
) -> tuple[list[CorpusRecord], dict[str, object]]:
    """Prepare the RAG corpus and write metadata plus validation reports."""
    base = base_dir or Path.cwd()
    input_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    documents = sorted(path for path in input_dir.iterdir() if path.is_file())
    if not documents:
        summary: dict[str, object] = {
            "blocked": True,
            "reason": "input directory contains no documents",
            "documents_found": 0,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
        }
        write_blocking_report(results_dir / "corpus_report.md", input_dir)
        return [], summary

    existing_metadata = load_existing_metadata(metadata_path)
    existing_ids = [
        row.get("document_id", "")
        for row in existing_metadata.values()
        if row.get("document_id", "")
    ]
    seen_hashes: dict[str, Path] = {}
    records: list[CorpusRecord] = []
    duplicate_count = 0

    for index, source in enumerate(documents, start=1):
        notes: list[str] = []
        sha256 = compute_sha256(source)
        suffix = source.suffix.lower()
        extracted: ExtractedDocument | None = None
        status = "accepted"

        if suffix not in SUPPORTED_EXTENSIONS:
            status = "rejected"
            notes.append("unsupported file type")
        elif sha256 in seen_hashes:
            status = "rejected"
            duplicate_count += 1
            notes.append(f"exact duplicate of {seen_hashes[sha256].name}")
        else:
            try:
                extracted = extract_document(source)
                if not extracted.text.strip():
                    status = "rejected"
                    notes.append("empty extracted text")
            except Exception as exc:  # noqa: BLE001 - report extraction failures as corpus validation data.
                status = "rejected"
                notes.append(f"text extraction failed: {exc}")

        destination_dir = accepted_dir if status == "accepted" else rejected_dir
        copied_path = copy_document(source, destination_dir)
        if status == "accepted":
            seen_hashes[sha256] = source

        existing = existing_metadata.get(sha256) or existing_metadata.get(
            normalize_relative(copied_path, base)
        ) or {}
        record = build_record(
            source=source,
            copied_path=copied_path,
            base_dir=base,
            sha256=sha256,
            extracted=extracted,
            status=status,
            notes=notes,
            existing=existing,
            document_id=next_document_id(existing_ids, index),
        )
        records.append(record)

    write_metadata(metadata_path, records)
    summary = write_results(results_dir, records, duplicate_count)
    return records, summary


def write_metadata(metadata_path: Path, records: list[CorpusRecord]) -> None:
    """Write the normalized document source metadata CSV."""
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())


def topic_distribution(records: list[CorpusRecord]) -> dict[str, int]:
    """Count accepted documents assigned to each required topic."""
    distribution = {topic: 0 for topic in REQUIRED_TOPICS}
    for record in records:
        if record.status != "accepted":
            continue
        for topic in [value.strip() for value in record.topics.split(";") if value.strip()]:
            if topic in distribution:
                distribution[topic] += 1
    return distribution


def missing_metadata(records: list[CorpusRecord]) -> dict[str, int]:
    """Count missing bibliographic metadata fields across accepted documents."""
    fields = ["title", "authors", "year", "organization", "source_url", "license"]
    accepted = [record for record in records if record.status == "accepted"]
    return {
        field: sum(1 for record in accepted if not getattr(record, field).strip())
        for field in fields
    }


def write_results(
    results_dir: Path,
    records: list[CorpusRecord],
    duplicate_count: int,
) -> dict[str, object]:
    """Write JSON, CSV and Markdown corpus validation outputs."""
    distribution = topic_distribution(records)
    missing_topics = [topic for topic, count in distribution.items() if count == 0]
    rejected = [record for record in records if record.status == "rejected"]
    accepted = [record for record in records if record.status == "accepted"]
    missing_bibliography = missing_metadata(records)
    summary: dict[str, object] = {
        "documents_found": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duplicates": duplicate_count,
        "coverage_by_topic": distribution,
        "missing_topics": missing_topics,
        "missing_metadata": missing_bibliography,
    }

    (results_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_distribution_csv(results_dir / "corpus_distribution.csv", distribution)
    write_rejected_csv(results_dir / "rejected_documents.csv", rejected)
    write_report(results_dir / "corpus_report.md", summary)
    return summary


def write_distribution_csv(path: Path, distribution: dict[str, int]) -> None:
    """Write topic coverage distribution as CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topic", "accepted_documents"])
        writer.writeheader()
        for topic, count in distribution.items():
            writer.writerow({"topic": topic, "accepted_documents": count})


def write_rejected_csv(path: Path, rejected: list[CorpusRecord]) -> None:
    """Write rejected document records and rejection notes."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["document_id", "local_path", "sha256", "file_type", "notes"],
        )
        writer.writeheader()
        for record in rejected:
            writer.writerow(
                {
                    "document_id": record.document_id,
                    "local_path": record.local_path,
                    "sha256": record.sha256,
                    "file_type": record.file_type,
                    "notes": record.notes,
                }
            )


def write_report(path: Path, summary: dict[str, object]) -> None:
    """Write a short Markdown report without copying source document content."""
    coverage = summary["coverage_by_topic"]
    assert isinstance(coverage, dict)
    missing_metadata_summary = summary["missing_metadata"]
    assert isinstance(missing_metadata_summary, dict)
    lines = [
        "# RAG Corpus Report",
        "",
        "## Counts",
        "",
        f"- Documents found: {summary['documents_found']}",
        f"- Accepted: {summary['accepted']}",
        f"- Rejected: {summary['rejected']}",
        f"- Exact duplicates: {summary['duplicates']}",
        "",
        "## Topic Coverage",
        "",
    ]
    for topic, count in coverage.items():
        lines.append(f"- {topic}: {count}")
    lines.extend(["", "## Missing Bibliographic Metadata", ""])
    for field, count in missing_metadata_summary.items():
        lines.append(f"- {field}: {count}")
    missing_topics = summary["missing_topics"]
    if missing_topics:
        lines.extend(["", "## Coverage Gaps", ""])
        for topic in missing_topics:
            lines.append(f"- {topic}")
    else:
        lines.extend(["", "Minimum topic coverage is complete."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocking_report(path: Path, input_dir: Path) -> None:
    """Write the only report produced when corpus preparation cannot start."""
    lines = [
        "# RAG Corpus Blocking Report",
        "",
        "Corpus preparation was blocked because no documents were found in the input directory.",
        "",
        f"- Input directory: {input_dir.as_posix()}",
        "- Documents found: 0",
        "- Action required: add real PDF, TXT, or Markdown documents to the inbox.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    records, summary = process_corpus(
        input_dir=args.input,
        accepted_dir=args.accepted,
        rejected_dir=args.rejected,
        metadata_path=args.metadata,
        results_dir=args.results,
    )
    if summary.get("blocked"):
        print("Blocked: no documents were available in the input directory.")
        print(f"Blocking report: {(args.results / 'corpus_report.md').as_posix()}")
        return 1

    print(f"Documents found: {summary['documents_found']}")
    print(f"Accepted: {summary['accepted']}")
    print(f"Rejected: {summary['rejected']}")
    print(f"Exact duplicates: {summary['duplicates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
