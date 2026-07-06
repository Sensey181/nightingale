"""Recherche vectorielle — encode la requête et interroge ChromaDB.

Responsabilité
--------------
Fine couche d'orchestration entre l'`EmbeddingModel` (encodage de la requête)
et le `VectorStore` (recherche de similarité). Isolée du reste pour que la
fusion RRF manipule des interfaces homogènes (`bm25.search` vs
`vector_search.search`).
"""

from __future__ import annotations

from indexing.embeddings import EmbeddingModel
from indexing.vector_store import VectorStore
from models import RetrievedChunk


class VectorSearch:
    """Recherche sémantique : requête → embedding → top-k ChromaDB."""

    def __init__(self, embedder: EmbeddingModel, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self, query: str, top_k: int, label: str | None = None
    ) -> list[RetrievedChunk]:
        """Retourne les `top_k` chunks sémantiquement proches de la requête.

        Si `label` est fourni, la recherche est restreinte aux chunks portant
        ce label via un filtre de métadonnées ChromaDB.
        """
        query_embedding = self.embedder.embed_query(query)
        if label:
            return self.store.query(
                query_embedding, top_k, where={"label": label}
            )
        return self.store.query(query_embedding, top_k)
