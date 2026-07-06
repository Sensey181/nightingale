"""CLI d'évaluation du retrieval : `python -m evaluation <eval_set.json>`.

Charge un jeu de questions annotées, exécute le pipeline de retrieval hybride
sur l'index existant et affiche MRR et NDCG@k.

Exemple :
    python -m evaluation data/eval/questions.json --k 5 --id-field chunk
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.harness import evaluate, load_eval_set
from retrieval.pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du retrieval ClinicalRAG")
    parser.add_argument("eval_set", type=Path, help="Jeu d'évaluation JSON.")
    parser.add_argument("--k", type=int, default=5, help="Profondeur NDCG@k.")
    parser.add_argument(
        "--id-field",
        choices=["chunk", "doc"],
        default="chunk",
        help="Identifiant comparé à la vérité terrain.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    eval_set = load_eval_set(args.eval_set)
    pipeline = build_pipeline()
    report = evaluate(pipeline, eval_set, k=args.k, id_field=args.id_field)

    print(f"Questions      : {report['n']}")
    print(f"MRR            : {report['mrr']:.4f}")
    print(f"NDCG@{report['k']:<9}: {report['ndcg@k']:.4f}")

    if args.verbose:
        print("\nDétail par question :")
        for row in report["per_query"]:
            print(
                f"  rr={row['rr']:.3f} ndcg={row['ndcg']:.3f}  {row['question']}"
            )


if __name__ == "__main__":
    main()
