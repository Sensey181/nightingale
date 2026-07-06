"""CLI de génération bout-à-bout : `python -m llm "ma question"`.

Enchaîne le retrieval hybride (BM25 + vectoriel + RRF + reranking) et la
génération Anthropic pour répondre à une question, réponse citée + liste des
sources utilisées. C'est le chemin complet du RAG en ligne de commande, avant
l'interface Streamlit.

Exemple :
    python -m llm "Quel est le motif d'hospitalisation ?" -v
"""

from __future__ import annotations

import argparse
import logging

from llm.anthropic_client import AnthropicClient
from retrieval.pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Génération RAG ClinicalRAG")
    parser.add_argument("query", help="Question médicale.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = build_pipeline()
    chunks = pipeline.retrieve(args.query)
    if not chunks:
        print("Aucun contexte trouvé (index vide ? lancer ingestion + indexation).")
        return

    answer = AnthropicClient().generate(args.query, chunks)
    print(answer)

    print("\n--- Sources ---")
    for i, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        print(f"[source {i}] {chunk.doc_id} · section={chunk.section.value}")


if __name__ == "__main__":
    main()
