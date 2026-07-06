"""Orchestration de l'indexing — point d'entrée public du module.

Enchaîne :
    RawDocument (data/processed)  ──▶  chunking  ──▶  embeddings (e5-small)
                                  ──▶  ChromaDB (upsert)

Le corpus BM25 (recherche lexicale) n'est pas construit ici : il est reconstruit
côté retrieval à partir de `VectorStore.get_all_chunks()`, sur le même corpus.

CLI : `python -m indexing` (voir `indexing/__main__.py`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from chunking.semantic_chunker import SemanticChunker
from config import Settings, get_settings
from indexing.embeddings import EmbeddingModel
from indexing.vector_store import VectorStore
from models import Chunk, RawDocument

logger = logging.getLogger(__name__)


def load_processed_documents(processed_dir: Path) -> list[RawDocument]:
    """Charge les `RawDocument` JSON produits par l'ingestion."""
    files = sorted(Path(processed_dir).glob("*.json"))
    return [
        RawDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in files
    ]


def index_documents(
    documents: list[RawDocument],
    settings: Settings | None = None,
    chunker: SemanticChunker | None = None,
    embedder: EmbeddingModel | None = None,
    store: VectorStore | None = None,
) -> int:
    """Chunke, encode et indexe une liste de documents dans ChromaDB.

    Args:
        documents: documents nettoyés à indexer.
        settings: configuration (chargée si non fournie).
        chunker / embedder / store: composants injectables (créés sinon).

    Returns:
        Le nombre de chunks indexés.
    """
    settings = settings or get_settings()
    chunker = chunker or SemanticChunker()
    embedder = embedder or EmbeddingModel(settings)
    store = store or VectorStore(settings)

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))

    if not chunks:
        logger.warning("Aucun chunk à indexer.")
        return 0

    logger.info("Encodage de %d chunks (e5-small, CPU)…", len(chunks))
    embeddings = embedder.embed_passages([c.text for c in chunks])

    logger.info("Indexation dans ChromaDB…")
    store.add_chunks(chunks, embeddings)
    logger.info("Indexation terminée : %d chunks.", len(chunks))
    return len(chunks)


def build_index(
    settings: Settings | None = None,
    processed_dir: Path | None = None,
    target_chars: int = 1000,
    overlap_chars: int = 150,
    reset: bool = False,
) -> int:
    """Construit l'index à partir des documents de `data/processed`.

    Args:
        settings: configuration (chargée si non fournie).
        processed_dir: répertoire des RawDocument (défaut : config).
        target_chars / overlap_chars: paramètres du chunker.
        reset: si vrai, vide la collection avant réindexation.

    Returns:
        Le nombre de chunks indexés durant cet appel.
    """
    settings = settings or get_settings()
    processed_dir = Path(processed_dir or settings.processed_dir)

    documents = load_processed_documents(processed_dir)
    if not documents:
        logger.warning(
            "Aucun RawDocument dans %s (lancer l'ingestion d'abord).", processed_dir
        )
        return 0

    store = VectorStore(settings)
    if reset:
        logger.info("Réinitialisation de la collection %s.", settings.chroma_collection)
        store.reset()

    return index_documents(
        documents,
        settings=settings,
        chunker=SemanticChunker(target_chars, overlap_chars),
        store=store,
    )
