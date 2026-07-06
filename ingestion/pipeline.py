"""Orchestration de l'ingestion — point d'entrée public du module.

Enchaîne, pour un fichier :
    ocr (MinerU local)  ──▶  parser (blocs + sections)  ──▶  cleaner  ──▶  RawDocument

Expose aussi une fonction batch qui ingère tous les fichiers d'un répertoire
(`data/raw`) et persiste les `RawDocument` en JSON dans `data/processed`.

Le CLI est accessible via `python -m ingestion` (voir `ingestion/__main__.py`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from config import Settings, get_settings
from ingestion.cleaner import clean_document
from ingestion.ocr import MinerUClient
from ingestion.parser import parse_markdown
from models import RawDocument

logger = logging.getLogger(__name__)

# Extensions de fichiers acceptées par l'OCR MinerU.
_SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
# Caractères non alphanumériques pour l'identifiant de document.
_DOC_ID_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _make_doc_id(file_path: Path) -> str:
    """Dérive un `doc_id` stable et sûr à partir du nom de fichier."""
    return _DOC_ID_RE.sub("_", file_path.stem).strip("_") or "document"


def ingest_file(
    file_path: Path,
    client: MinerUClient | None = None,
) -> RawDocument:
    """Ingère un fichier unique : OCR → parsing → nettoyage.

    Args:
        file_path: chemin du fichier source (PDF/image).
        client: client MinerU réutilisable (créé si non fourni).

    Returns:
        Le `RawDocument` nettoyé.
    """
    file_path = Path(file_path)
    client = client or MinerUClient()

    logger.info("Ingestion : %s", file_path.name)
    markdown = client.ocr_to_markdown(file_path)
    blocks = parse_markdown(markdown)
    document = clean_document(
        doc_id=_make_doc_id(file_path),
        source_path=str(file_path),
        blocks=blocks,
    )
    logger.info(
        "Ingestion terminée : %s (%d sections, %d caractères)",
        document.doc_id,
        len(document.metadata.get("sections", [])),
        len(document.text),
    )
    return document


def ingest_directory(
    settings: Settings | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> list[RawDocument]:
    """Ingère tous les fichiers supportés d'un répertoire et les persiste.

    Chaque `RawDocument` est écrit en JSON dans `processed_dir` sous
    `<doc_id>.json`. Les fichiers déjà traités sont ignorés (idempotent).

    Args:
        settings: configuration (chargée si non fournie).
        raw_dir: répertoire des fichiers bruts (défaut : `settings.raw_dir`).
        processed_dir: répertoire de sortie (défaut : `settings.processed_dir`).

    Returns:
        La liste des `RawDocument` produits durant cet appel.
    """
    settings = settings or get_settings()
    raw_dir = Path(raw_dir or settings.raw_dir)
    processed_dir = Path(processed_dir or settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    if not files:
        logger.warning("Aucun fichier supporté trouvé dans %s", raw_dir)
        return []

    client = MinerUClient(settings)
    documents: list[RawDocument] = []
    for file_path in files:
        out_path = processed_dir / f"{_make_doc_id(file_path)}.json"
        if out_path.exists():
            logger.info("Déjà ingéré, ignoré : %s", out_path.name)
            continue
        document = ingest_file(file_path, client=client)
        out_path.write_text(
            document.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("Écrit : %s", out_path)
        documents.append(document)

    return documents
