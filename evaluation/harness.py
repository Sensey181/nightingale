"""Harnais d'évaluation du retrieval sur un jeu de questions annotées.

Exécute le `RetrievalPipeline` sur chaque question d'un jeu d'évaluation et
agrège les métriques (MRR, NDCG@k). La qualité du retrieval conditionnant toute
la qualité du RAG, on évalue cette brique isolément (avant la génération).

Jeu d'évaluation (JSON)
-----------------------
    [
      {"question": "...", "relevant_ids": ["doc1::0003", ...]},
      ...
    ]

`id_field` choisit l'identifiant comparé :
- "chunk" (défaut) : compare les `chunk_id` (annotation fine) ;
- "doc"           : compare les `doc_id` (annotation au niveau document).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from evaluation.metrics import (
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank,
)
from models import RetrievedChunk


class _Retriever(Protocol):
    """Interface minimale attendue (RetrievalPipeline ou doublure de test)."""

    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


def load_eval_set(path: Path) -> list[dict]:
    """Charge un jeu d'évaluation JSON (`[{question, relevant_ids}, ...]`)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Le jeu d'évaluation doit être une liste d'items.")
    return data


def _retrieved_ids(results: list[RetrievedChunk], id_field: str) -> list[str]:
    if id_field == "doc":
        # Déduplique les doc_id en conservant l'ordre d'apparition.
        seen: dict[str, None] = {}
        for r in results:
            seen.setdefault(r.chunk.doc_id, None)
        return list(seen)
    return [r.chunk.chunk_id for r in results]


def evaluate(
    retriever: _Retriever,
    eval_set: list[dict],
    k: int = 5,
    id_field: str = "chunk",
) -> dict:
    """Évalue le retrieval sur un jeu de questions annotées.

    Args:
        retriever: objet exposant `retrieve(query) -> list[RetrievedChunk]`.
        eval_set: liste d'items `{question, relevant_ids}`.
        k: profondeur pour le NDCG@k.
        id_field: "chunk" (défaut) ou "doc".

    Returns:
        Un dict avec les métriques agrégées et le détail par question :
        `{"n", "k", "mrr", "ndcg@k", "per_query": [...]}`.
    """
    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    per_query: list[dict] = []

    for item in eval_set:
        question = item["question"]
        relevant = set(item.get("relevant_ids", []))
        retrieved = _retrieved_ids(retriever.retrieve(question), id_field)

        all_retrieved.append(retrieved)
        all_relevant.append(relevant)
        per_query.append(
            {
                "question": question,
                "rr": reciprocal_rank(retrieved, relevant),
                "ndcg": ndcg_at_k(retrieved, relevant, k),
            }
        )

    return {
        "n": len(eval_set),
        "k": k,
        "mrr": mean_reciprocal_rank(all_retrieved, all_relevant),
        "ndcg@k": mean_ndcg_at_k(all_retrieved, all_relevant, k),
        "per_query": per_query,
    }
