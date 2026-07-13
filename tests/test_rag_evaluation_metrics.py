from scripts.evaluate_rag import first_relevant_rank, hit_at_k, reciprocal_rank


def test_hit_at_k_returns_one_when_expected_document_is_within_k() -> None:
    ranked = ["DOC009", "DOC002", "DOC001"]
    expected = ["DOC001", "DOC004"]

    assert hit_at_k(ranked, expected, 1) == 0.0
    assert hit_at_k(ranked, expected, 3) == 1.0


def test_hit_at_k_returns_zero_without_expected_documents() -> None:
    ranked = ["DOC003", "DOC002", "DOC006"]

    assert hit_at_k(ranked, ["DOC001"], 5) == 0.0
    assert hit_at_k(ranked, [], 5) == 0.0
    assert hit_at_k(ranked, ["DOC003"], 0) == 0.0


def test_first_relevant_rank_and_mrr_use_first_matching_rank() -> None:
    ranked = ["DOC004", "DOC002", "DOC001", "DOC003"]
    expected = ["DOC001", "DOC003"]

    assert first_relevant_rank(ranked, expected) == 3
    assert reciprocal_rank(ranked, expected) == 1 / 3


def test_mrr_is_zero_when_no_expected_document_is_retrieved() -> None:
    assert first_relevant_rank(["DOC004", "DOC002"], ["DOC001"]) is None
    assert reciprocal_rank(["DOC004", "DOC002"], ["DOC001"]) == 0.0
