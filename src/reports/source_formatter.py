from __future__ import annotations


def format_sources(items: list[dict]) -> list[str]:
    formatted: list[str] = []
    for index, item in enumerate(items, start=1):
        title = item.get("title", "Sin título")
        institution = item.get("author_or_institution", "Fuente no indicada")
        year = item.get("year", "s. f.")
        formatted.append(f"[{index}] {institution} ({year}). {title}.")
    return formatted
