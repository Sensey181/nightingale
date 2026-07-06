"""CLI d'ingestion : `python -m ingestion`.

Ingère tous les fichiers supportés de `data/raw` (ou d'un répertoire fourni)
via l'OCR MinerU, puis persiste les `RawDocument` en JSON dans `data/processed`.

Exemples :
    python -m ingestion
    python -m ingestion --raw-dir ./data/raw --processed-dir ./data/processed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_settings
from ingestion.pipeline import ingest_directory


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingestion OCR ClinicalRAG")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=settings.raw_dir,
        help="Répertoire des fichiers bruts (PDF/images).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=settings.processed_dir,
        help="Répertoire de sortie des RawDocument JSON.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Logs détaillés."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    documents = ingest_directory(
        settings=settings,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    print(f"{len(documents)} document(s) ingéré(s) dans {args.processed_dir}")


if __name__ == "__main__":
    main()
