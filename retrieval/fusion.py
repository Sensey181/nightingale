"""Fusion des résultats hybrides par Reciprocal Rank Fusion (RRF).

Principe
--------
RRF combine plusieurs listes classées (ici BM25 et vectoriel) sans avoir à
calibrer leurs scores, souvent d'échelles incomparables. Chaque document reçoit
un score fondé sur ses *rangs* :

        score(d) = Σ_listes  1 / (k + rang_liste(d))

où `k` (constante RRF, ~60 par défaut) amortit l'influence des tout premiers
rangs et rend la fusion robuste. Les documents bien classés dans plusieurs
listes remontent naturellement.

Pourquoi RRF ici
----------------
Les scores BM25 (non bornés) et la similarité cosinus (dans [-1, 1]) ne sont pas
directement comparables. RRF s'appuie uniquement sur les rangs → fusion simple,
robuste et sans réglage fin.
"""

from __future__ import annotations

from models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fusionne plusieurs listes classées via RRF.

    Args:
        ranked_lists: listes de résultats (ex. [résultats_bm25, résultats_vect]),
            chacune ordonnée par pertinence décroissante.
        k: constante RRF.

    Returns:
        Une liste unique de `RetrievedChunk` triée par score RRF décroissant,
        dédupliquée par `chunk_id`.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for results in ranked_lists:
        for rank, retrieved in enumerate(results, start=1):
            chunk_id = retrieved.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, retrieved.chunk)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        RetrievedChunk(
            chunk=chunks[chunk_id], score=score, retrieval_source="rrf"
        )
        for chunk_id, score in ordered
    ]
