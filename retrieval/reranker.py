"""Reranking des candidats fusionnés avec un cross-encoder (CPU).

Responsabilité
--------------
Ré-ordonner finement les candidats issus de la fusion RRF avec le cross-encoder
`ms-marco-MiniLM-L6-v2`, puis ne conserver que les `top_n` meilleurs à
transmettre au LLM.

Bi-encoder vs cross-encoder
---------------------------
- La 1re phase (embeddings e5 + BM25) est rapide et scalable mais approximative.
- Le cross-encoder lit la paire (requête, chunk) conjointement et fournit un
  score de pertinence bien plus précis — au prix d'un coût par paire. On ne
  l'applique donc qu'aux candidats déjà pré-filtrés par la fusion, ce qui reste
  tenable sur CPU.

Ce reranking est le dernier maillon local avant la génération : réduire le
contexte aux `top_n` chunks vraiment pertinents améliore la qualité de la
réponse et limite les tokens envoyés au LLM. L'import du modèle est paresseux.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings
from models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder ms-marco-MiniLM-L6-v2 pour le reranking final (CPU)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None  # CrossEncoder, chargé paresseusement

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Chargement du reranker %s (CPU)", self.settings.reranker_model
            )
            self._model = CrossEncoder(self.settings.reranker_model, device="cpu")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Ré-ordonne les candidats et retourne les `top_n` meilleurs."""
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk=candidate.chunk,
                score=float(score),
                retrieval_source="reranked",
            )
            for candidate, score in ranked[:top_n]
        ]
