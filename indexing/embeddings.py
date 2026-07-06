"""Génération d'embeddings locaux avec multilingual-e5-small (CPU).

Responsabilité
--------------
Encoder des textes (chunks à l'indexation, requêtes au retrieval) en vecteurs
denses **normalisés**, en local sur CPU.

Détails propres à e5
--------------------
Les modèles E5 attendent des préfixes asymétriques :
- `"passage: "` pour les documents indexés ;
- `"query: "` pour les requêtes.
Cette convention est gérée ici afin que le reste du pipeline manipule du texte
brut. Les vecteurs sont normalisés (L2) pour que la similarité cosinus utilisée
par ChromaDB soit cohérente.

Le `SentenceTransformer` est chargé une seule fois (import et chargement
paresseux : le module s'importe sans `torch`/`sentence-transformers` installés,
ce qui garde le pipeline importable et testable).
"""

from __future__ import annotations

import logging

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


class EmbeddingModel:
    """Encapsule le chargement et l'inférence de multilingual-e5-small (CPU)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None  # SentenceTransformer, chargé paresseusement

    def _ensure_model(self):
        """Charge le modèle au premier usage (import paresseux)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Chargement du modèle d'embedding %s (CPU)",
                self.settings.embedding_model,
            )
            self._model = SentenceTransformer(
                self.settings.embedding_model, device="cpu"
            )
        return self._model

    def embed_passages(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Encode des passages (documents) — préfixe 'passage: ', normalisé."""
        if not texts:
            return []
        model = self._ensure_model()
        prefixed = [f"{_PASSAGE_PREFIX}{text}" for text in texts]
        embeddings = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode une requête — préfixe 'query: ', normalisé."""
        model = self._ensure_model()
        embedding = model.encode(
            f"{_QUERY_PREFIX}{text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
