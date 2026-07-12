from src.rag.prompt_builder import build_retrieval_query


def test_spotted_query_is_cautious() -> None:
    query = build_retrieval_query("spotted", "manchas oscuras")
    assert "posibles causas" in query
    assert "manchas oscuras" in query
