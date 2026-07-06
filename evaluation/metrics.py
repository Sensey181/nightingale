"""Métriques de retrieval : MRR et NDCG.

Ces métriques comparent la liste ordonnée d'identifiants retournée par le
retrieval aux identifiants jugés pertinents (vérité terrain).

Format de la vérité terrain
---------------------------
Une évaluation est une liste d'items :
    {
        "question": "...",
        "relevant_ids": ["chunk_id_1", "chunk_id_2", ...]  # ou doc_id
    }

Le harnais (`evaluation/harness.py`) exécute le retrieval sur chaque question,
puis calcule les métriques ci-dessous sur les `retrieved_ids` (ordonnés) vs
`relevant_ids`.

Implémentation volontairement sans dépendance (juste `math.log2`) : les
métriques sont testables sans numpy ni modèle.
"""

from __future__ import annotations

from math import log2


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Rang réciproque : 1 / rang du premier document pertinent (0 si aucun)."""
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """NDCG@k avec gains binaires (1 si pertinent, 0 sinon), normalisé par l'IDCG.

    DCG@k  = Σ_{i=1..k}  rel_i / log2(i + 1)
    IDCG@k = DCG idéal (tous les pertinents remontés en tête)
    NDCG@k = DCG@k / IDCG@k   (0 si IDCG nul)
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / log2(i + 1)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(
    all_retrieved: list[list[str]], all_relevant: list[set[str]]
) -> float:
    """MRR : moyenne des rangs réciproques sur l'ensemble des questions."""
    if not all_retrieved:
        return 0.0
    total = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    )
    return total / len(all_retrieved)


def mean_ndcg_at_k(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
    k: int,
) -> float:
    """NDCG@k moyen sur l'ensemble des questions."""
    if not all_retrieved:
        return 0.0
    total = sum(
        ndcg_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    )
    return total / len(all_retrieved)
