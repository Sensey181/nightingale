"""Orchestration du retrieval hybride — point d'entrée public du module.

Enchaîne les étapes :
    1. BM25          (lexical)      → top_k_bm25 candidats
    2. VectorSearch  (sémantique)   → top_k_vector candidats
    3. RRF           (fusion)       → liste fusionnée dédupliquée
    4. Reranker      (cross-encoder) → top_n_rerank chunks finaux

Tout est local (aucun appel réseau) → cœur de la conformité HDS côté données.
Les paramètres (top_k, k RRF, top_n) proviennent de la configuration centrale.

`build_pipeline()` assemble le pipeline complet depuis la config (c'est le point
d'entrée utilisé par l'app Streamlit).
"""

from __future__ import annotations

from config import Settings, get_settings
from indexing.embeddings import EmbeddingModel
from indexing.vector_store import VectorStore
from models import RetrievedChunk
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from retrieval.vector_search import VectorSearch


class RetrievalPipeline:
    """Pipeline de recherche hybride BM25 + vectoriel, fusion RRF, reranking."""

    def __init__(
        self,
        bm25: BM25Index,
        vector_search: VectorSearch,
        reranker: Reranker,
        settings: Settings | None = None,
    ) -> None:
        self.bm25 = bm25
        self.vector_search = vector_search
        self.reranker = reranker
        self.settings = settings or get_settings()

    def retrieve(
        self, query: str, label: str | None = None
    ) -> list[RetrievedChunk]:
        """Exécute le retrieval hybride complet pour une requête.

        Args:
            query: question médicale de l'utilisateur.
            label: si fourni, restreint la recherche (BM25 + vectoriel) aux
                seuls chunks portant ce label.

        Returns:
            Les `top_n_rerank` chunks les plus pertinents, ordonnés.
        """
        bm25_results = self.bm25.search(
            query, self.settings.top_k_bm25, label=label
        )
        vector_results = self.vector_search.search(
            query, self.settings.top_k_vector, label=label
        )
        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results], k=self.settings.rrf_k
        )
        return self.reranker.rerank(query, fused, self.settings.top_n_rerank)


def build_pipeline(settings: Settings | None = None) -> RetrievalPipeline:
    """Assemble le pipeline de retrieval depuis la configuration.

    Charge l'index vectoriel (ChromaDB), reconstruit l'index BM25 à partir du
    même corpus (`VectorStore.get_all_chunks()`), et instancie le reranker.
    """
    settings = settings or get_settings()
    embedder = EmbeddingModel(settings)
    store = VectorStore(settings)
    bm25 = BM25Index(store.get_all_chunks())
    vector_search = VectorSearch(embedder, store)
    reranker = Reranker(settings)
    return RetrievalPipeline(bm25, vector_search, reranker, settings)
