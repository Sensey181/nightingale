"""Module de retrieval — recherche hybride, fusion RRF et reranking.

Rôle dans le pipeline
---------------------
    requête  ──▶  bm25.py          (recherche lexicale)      ┐
             ──▶  vector_search.py (recherche vectorielle)   � ─▶ fusion.py (RRF)
                                                             ┘         │
                                                                       ▼
                                                       reranker.py (cross-encoder)
                                                                       │
                                                                       ▼
                                                            list[RetrievedChunk]

`pipeline.py` orchestre l'ensemble et constitue le point d'entrée public du
module.

Choix d'architecture
--------------------
- **Recherche hybride BM25 + vectoriel** : le lexical (BM25) capture les termes
  médicaux exacts (noms de molécules, codes, valeurs) que l'embedding peut
  lisser ; le vectoriel capture la similarité sémantique. Les deux sont
  complémentaires sur du texte clinique.
- **Fusion RRF (Reciprocal Rank Fusion)** : fusion robuste et sans calibration
  d'échelle, qui combine les rangs des deux listes plutôt que leurs scores
  hétérogènes.
- **Reranking (ms-marco-MiniLM-L6-v2, CPU)** : un cross-encoder ré-ordonne
  finement les candidats fusionnés pour ne garder que les `top_n` les plus
  pertinents transmis au LLM. Tourne en local, sans GPU.

Tout le retrieval est **100 % local** (aucun appel réseau) → cœur de la
conformité HDS côté données.
"""
