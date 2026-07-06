"""Module de chunking — segmentation sémantique adaptée aux CR cliniques.

Rôle dans le pipeline
---------------------
    RawDocument  ──▶  semantic_chunker.py  ──▶  list[Chunk]

Choix d'architecture
--------------------
Plutôt qu'un découpage aveugle à taille fixe, le chunking s'appuie sur la
structure clinique détectée à l'ingestion (motif d'hospitalisation,
antécédents, biologie, conclusion…). Chaque `Chunk` :
- reste dans les limites d'une section (pas de mélange biologie/conclusion) ;
- porte son `ClinicalSection` en métadonnée, exploitable au retrieval ;
- respecte une taille cible (avec léger chevauchement) pour préserver le
  contexte sans exploser le nombre de tokens à embedder.

Cet alignement section ↔ chunk est le principal levier de qualité du retrieval
sur des questions médicales ciblées.
"""
