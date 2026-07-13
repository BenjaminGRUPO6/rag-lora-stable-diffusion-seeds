from __future__ import annotations


def format_sources(items: list[dict]) -> list[str]:
    """Format retrieved source metadata as numbered references."""
    formatted: list[str] = []
    for index, item in enumerate(items, start=1):
        title = str(item.get("title") or item.get("document_title") or "Sin titulo")
        institution = str(
            item.get("author_or_institution")
            or item.get("organization")
            or item.get("document_id")
            or "Fuente no indicada"
        )
        year = str(item.get("year") or "s. f.")
        locator = str(item.get("source_url") or item.get("source") or item.get("local_path") or "")
        page = item.get("page")
        page_text = f", p. {page}" if page is not None else ""
        locator_text = f" {locator}" if locator else ""
        formatted.append(f"[{index}] {institution} ({year}). {title}{page_text}.{locator_text}".strip())
    return formatted
