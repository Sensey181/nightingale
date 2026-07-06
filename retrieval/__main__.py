"""CLI de retrieval : `python -m retrieval "ma question"`.

Assemble le pipeline hybride (BM25 + vectoriel + RRF + reranking) depuis l'index
ChromaDB existant, exécute une requête et affiche les chunks retenus (score,
section, extrait). Utile pour vérifier la qualité du retrieval avant la
génération.

Exemple :
    python -m retrieval "Quel est le motif d'hospitalisation ?" -v
"""

from __future__ import annotations

import argparse
import logging

from retrieval.pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval hybride ClinicalRAG")
    parser.add_argument("query", help="Question médicale.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = build_pipeline()
    results = pipeline.retrieve(args.query)

    if not results:
        print("Aucun résultat (index vide ? lancer l'ingestion + l'indexation).")
        return

    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        snippet = " ".join(chunk.text.split())[:200]
        print(
            f"[{i}] score={result.score:.4f} "
            f"section={chunk.section.value} doc={chunk.doc_id}"
        )
        print(f"    {snippet}\n")


if __name__ == "__main__":
    main()
