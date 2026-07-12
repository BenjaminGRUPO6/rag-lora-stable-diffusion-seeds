from __future__ import annotations

from src.rag.prompt_builder import build_report_prompt
from src.rag.retrieval import build_retrieval_query
from src.reports.report_generator import generate_template_report


def analyze_seed(prediction: dict, retriever, observations: list[str] | None = None) -> dict:
    query = build_retrieval_query(prediction, observations)
    retrieved = retriever(query)
    prompt = build_report_prompt(prediction, retrieved)
    report = generate_template_report(prediction, retrieved)
    return {"query": query, "retrieved": retrieved, "prompt": prompt, "report": report}
