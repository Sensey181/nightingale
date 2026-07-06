"""CLI d'inspection du chunking : `python -m chunking`.

Charge les `RawDocument` déjà ingérés (`data/processed/*.json`), applique le
chunker sémantique et affiche des statistiques (nombre de chunks, répartition
par section, tailles). Utile pour régler `target_chars` / `overlap_chars` avant
l'indexation. N'écrit rien : la persistance des chunks relève de l'indexing.

Exemples :
    python -m chunking
    python -m chunking --processed-dir ./data/processed --target 800 --overlap 120
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from chunking.semantic_chunker import SemanticChunker
from config import get_settings
from models import RawDocument


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Inspection du chunking ClinicalRAG")
    parser.add_argument("--processed-dir", type=Path, default=settings.processed_dir)
    parser.add_argument("--target", type=int, default=1000, help="target_chars")
    parser.add_argument("--overlap", type=int, default=150, help="overlap_chars")
    args = parser.parse_args()

    chunker = SemanticChunker(args.target, args.overlap)
    files = sorted(Path(args.processed_dir).glob("*.json"))
    if not files:
        print(f"Aucun RawDocument dans {args.processed_dir} (lancer l'ingestion).")
        return

    total = 0
    for path in files:
        document = RawDocument.model_validate_json(path.read_text(encoding="utf-8"))
        chunks = chunker.chunk(document)
        total += len(chunks)
        by_section = Counter(c.section.value for c in chunks)
        sizes = [len(c.text) for c in chunks]
        avg = sum(sizes) // len(sizes) if sizes else 0
        print(
            f"{document.doc_id}: {len(chunks)} chunks "
            f"(taille moy. {avg}, max {max(sizes, default=0)})"
        )
        for section, count in by_section.most_common():
            print(f"    {section:24s} {count}")

    print(f"\nTotal : {total} chunks sur {len(files)} document(s).")


if __name__ == "__main__":
    main()
