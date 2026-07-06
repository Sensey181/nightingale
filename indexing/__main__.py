"""CLI d'indexation : `python -m indexing`.

Charge les `RawDocument` de `data/processed`, les chunke, encode les chunks avec
multilingual-e5-small (CPU) et les indexe dans ChromaDB (`data/chroma`).

Exemples :
    python -m indexing
    python -m indexing --reset --target 800 --overlap 120 -v
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_settings
from indexing.pipeline import build_index
from indexing.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Indexation ClinicalRAG")
    parser.add_argument("--processed-dir", type=Path, default=settings.processed_dir)
    parser.add_argument("--target", type=int, default=1000, help="target_chars")
    parser.add_argument("--overlap", type=int, default=150, help="overlap_chars")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Vide la collection avant réindexation.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    n = build_index(
        settings=settings,
        processed_dir=args.processed_dir,
        target_chars=args.target,
        overlap_chars=args.overlap,
        reset=args.reset,
    )
    total = VectorStore(settings).count()
    print(f"{n} chunk(s) indexé(s) — collection totale : {total}")


if __name__ == "__main__":
    main()
