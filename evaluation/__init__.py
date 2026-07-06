"""Module d'évaluation — métriques de retrieval sur questions médicales.

Rôle
----
Mesurer la qualité du retrieval (avant génération), qui conditionne toute la
qualité du RAG. On évalue la capacité du pipeline à remonter les bons chunks
pour un jeu de questions médicales annotées.

Métriques
---------
- **MRR (Mean Reciprocal Rank)** : privilégie la position du premier document
  pertinent — pertinent quand une seule bonne source suffit.
- **NDCG (Normalized Discounted Cumulative Gain)** : récompense l'ordre global,
  en pondérant les positions — pertinent quand plusieurs sources comptent.

Jeu d'évaluation
----------------
Un fichier de questions annotées (question → chunk_id / doc_id pertinents),
construit sur MIMIC-IV, sert de vérité terrain. Le format est décrit dans
`evaluation/metrics.py`.
"""
