"""Tests des métriques de retrieval et du harnais d'évaluation.

Logique pure (pas de dépendance lourde). Le harnais est testé avec un
retriever factice qui renvoie une liste préétablie de `RetrievedChunk`.
"""

from __future__ import annotations

from math import log2

import pytest

from evaluation.harness import evaluate
from evaluation.metrics import (
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank,
)
from models import Chunk, ClinicalSection, RetrievedChunk


# --------------------------------------------------------------------------- #
# reciprocal_rank                                                             #
# --------------------------------------------------------------------------- #
def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(1 / 2)
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
    assert reciprocal_rank(["a", "b"], set()) == 0.0  # aucun pertinent


# --------------------------------------------------------------------------- #
# ndcg_at_k                                                                   #
# --------------------------------------------------------------------------- #
def test_ndcg_pertinent_en_tete_vaut_1() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)


def test_ndcg_position_2() -> None:
    # Un seul pertinent, au rang 2 : DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    expected = (1 / log2(3)) / 1.0
    assert ndcg_at_k(["a", "b", "c"], {"b"}, k=3) == pytest.approx(expected)


def test_ndcg_hors_k_est_nul() -> None:
    # Pertinent au rang 3 mais k=2 → non compté.
    assert ndcg_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_ndcg_deux_pertinents_normalise() -> None:
    # Pertinents remontés aux rangs 1 et 2 → NDCG parfait = 1.
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_cas_limites() -> None:
    assert ndcg_at_k(["a"], set(), k=3) == 0.0
    assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0


# --------------------------------------------------------------------------- #
# moyennes                                                                    #
# --------------------------------------------------------------------------- #
def test_moyennes() -> None:
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"y"}]
    # RR : 1.0 et 0.5 → MRR 0.75.
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == pytest.approx(0.75)
    # NDCG : 1.0 et 1/log2(3) → moyenne.
    expected = (1.0 + 1 / log2(3)) / 2
    assert mean_ndcg_at_k(all_retrieved, all_relevant, k=2) == pytest.approx(expected)
    assert mean_reciprocal_rank([], []) == 0.0


# --------------------------------------------------------------------------- #
# harnais                                                                     #
# --------------------------------------------------------------------------- #
def _rc(cid: str, doc_id: str = "doc") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        text="t",
        section=ClinicalSection.AUTRE,
        position=0,
    )
    return RetrievedChunk(chunk=chunk, score=1.0, retrieval_source="reranked")


class _FakeRetriever:
    def __init__(self, mapping: dict[str, list[RetrievedChunk]]) -> None:
        self.mapping = mapping

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.mapping[query]


def test_evaluate_chunk_level() -> None:
    retriever = _FakeRetriever(
        {
            "q1": [_rc("c1"), _rc("c2")],  # pertinent c1 au rang 1
            "q2": [_rc("c3"), _rc("c4")],  # pertinent c4 au rang 2
        }
    )
    eval_set = [
        {"question": "q1", "relevant_ids": ["c1"]},
        {"question": "q2", "relevant_ids": ["c4"]},
    ]
    report = evaluate(retriever, eval_set, k=2)

    assert report["n"] == 2
    assert report["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert len(report["per_query"]) == 2


def test_evaluate_doc_level_dedupe() -> None:
    # Deux chunks du même doc → un seul identifiant doc, au rang 1.
    retriever = _FakeRetriever(
        {"q": [_rc("c1", "docA"), _rc("c2", "docA"), _rc("c3", "docB")]}
    )
    report = evaluate(
        retriever, [{"question": "q", "relevant_ids": ["docA"]}], k=3, id_field="doc"
    )
    assert report["mrr"] == pytest.approx(1.0)
