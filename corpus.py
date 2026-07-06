"""Gestion du corpus au niveau document, pour la couche applicative (Streamlit).

Rôle
----
Composer ingestion + indexing + `VectorStore` en opérations « document » simples
à appeler depuis l'UI : lister, ajouter (OCR → index), supprimer, (re)labelliser.
Le retrieval ne dépend pas de ce module ; il reste cantonné à la lecture de
l'index.

Cohérence de la persistance
---------------------------
La base vectorielle (ChromaDB) est la source de vérité pour la recherche, mais
les `RawDocument` JSON de `data/processed` restent la source d'une éventuelle
réindexation complète. Les opérations qui modifient un document (suppression,
label) répercutent donc le changement **des deux côtés** :

    - suppression : chunks Chroma + `data/processed/<doc_id>.json` (le fichier
      brut d'origine est conservé pour permettre une ré-ingestion ultérieure) ;
    - label : métadonnées Chroma (effet immédiat sur le filtre de recherche) +
      champ `label` du JSON `RawDocument` (pour survivre à une réindexation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from config import Settings, get_settings
from indexing.pipeline import index_documents
from indexing.vector_store import VectorStore
from ingestion.pipeline import ingest_file
from models import RawDocument

logger = logging.getLogger(__name__)


def _processed_path(settings: Settings, doc_id: str) -> Path:
    return Path(settings.processed_dir) / f"{doc_id}.json"


def list_documents(store: VectorStore) -> list[dict]:
    """Liste les documents indexés (délégué au store)."""
    return store.list_documents()


def add_document(
    file_path: Path | str,
    *,
    label: str = "",
    settings: Settings | None = None,
    store: VectorStore | None = None,
    on_step: Callable[[str], None] | None = None,
) -> tuple[str, int]:
    """Ingère (OCR local) puis indexe un fichier, et le persiste.

    Args:
        file_path: fichier source déjà présent sur disque (idéalement dans
            `data/raw` pour permettre une réindexation ultérieure).
        label: étiquette optionnelle appliquée au document.
        settings / store: injectables (créés sinon).
        on_step: callback optionnel appelé avec un message à chaque phase
            (lecture/OCR, sauvegarde, indexation). Permet à l'UI d'afficher la
            progression au fil de l'eau.

    Returns:
        `(doc_id, n_chunks)` : identifiant du document et nombre de chunks
        indexés.

    Note:
        L'OCR (MinerU, local, CPU) peut être lent : à appeler avec un indicateur
        de progression côté UI.
    """
    settings = settings or get_settings()
    store = store or VectorStore(settings)

    def _step(message: str) -> None:
        if on_step is not None:
            on_step(message)

    _step("Lecture et OCR du document (MinerU, local)…")
    document = ingest_file(Path(file_path))
    if label:
        document.label = label
    _step(f"OCR terminé — document reconnu : {document.doc_id}")

    processed_dir = Path(settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _step("Sauvegarde du document traité…")
    _processed_path(settings, document.doc_id).write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )

    _step("Indexation (embeddings vectoriels + BM25)…")
    n_chunks = index_documents([document], settings=settings, store=store)
    _step(f"Indexation terminée — {n_chunks} chunks")
    logger.info("Document ajouté : %s (%d chunks)", document.doc_id, n_chunks)
    return document.doc_id, n_chunks


def delete_document(
    doc_id: str,
    *,
    settings: Settings | None = None,
    store: VectorStore | None = None,
) -> int:
    """Supprime un document de l'index et son `RawDocument` persisté.

    Le fichier brut d'origine (`data/raw`) est conservé. Retourne le nombre de
    chunks supprimés de l'index.
    """
    settings = settings or get_settings()
    store = store or VectorStore(settings)

    n_deleted = store.delete_document(doc_id)
    _processed_path(settings, doc_id).unlink(missing_ok=True)
    logger.info("Document supprimé : %s (%d chunks)", doc_id, n_deleted)
    return n_deleted


def set_document_label(
    doc_id: str,
    label: str,
    *,
    settings: Settings | None = None,
    store: VectorStore | None = None,
) -> int:
    """(Re)labellise un document dans l'index et son `RawDocument` persisté.

    Retourne le nombre de chunks mis à jour dans l'index.
    """
    settings = settings or get_settings()
    store = store or VectorStore(settings)

    n_updated = store.set_label(doc_id, label)

    processed = _processed_path(settings, doc_id)
    if processed.exists():
        document = RawDocument.model_validate_json(
            processed.read_text(encoding="utf-8")
        )
        document.label = label or ""
        processed.write_text(
            document.model_dump_json(indent=2), encoding="utf-8"
        )
    logger.info("Label de %s → %r (%d chunks)", doc_id, label, n_updated)
    return n_updated
