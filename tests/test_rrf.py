from app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_preserves_high_ranked_documents():
    results_a = [{"id": "d1"}, {"id": "d2"}]
    results_b = [{"id": "d3"}, {"id": "d1"}]

    fused = reciprocal_rank_fusion([results_a, results_b], k=60)
    # d1 appears high in both lists and should be present
    ids = [d["id"] for d in fused]
    assert "d1" in ids


def test_rrf_combines_duplicate_documents():
    results_a = [{"id": "x"}, {"id": "y"}]
    results_b = [{"id": "y"}, {"id": "x"}]

    fused = reciprocal_rank_fusion([results_a, results_b], k=60)
    ids = [d["id"] for d in fused]
    assert ids[0] in ("x", "y")
