"""Module d'indexing — embeddings locaux + stockage ChromaDB.

Rôle dans le pipeline
---------------------
    list[Chunk]  ──▶  embeddings.py    (multilingual-e5-small, CPU)
                 ──▶  vector_store.py  (persistance ChromaDB locale)

Choix d'architecture
--------------------
- **Embeddings locaux** : `multilingual-e5-small` tourne sur CPU, léger et
  multilingue (adapté au français clinique). Aucune donnée n'est envoyée à un
  service d'embedding externe → cohérent avec la contrainte « retrieval local ».
- **ChromaDB local** : base vectorielle persistée sur disque (`data/chroma`).
  Tout l'index reste sur la machine → conforme aux exigences HDS côté données.

Le corpus BM25 (recherche lexicale) est construit côté `retrieval/` à partir
des mêmes chunks ; l'indexing se concentre ici sur la partie vectorielle.
"""
