"""Module d'ingestion — OCR, parsing et nettoyage des comptes-rendus cliniques.

Rôle dans le pipeline
---------------------
Point d'entrée du pipeline : transforme des fichiers bruts (PDF scannés,
lettres de sortie) en `RawDocument` propres et prêts pour le chunking.

    fichier PDF/image  ──▶  ocr.py       (MinerU local → Markdown structuré)
                        ──▶  parser.py    (Markdown → texte + structure)
                        ──▶  cleaner.py   (normalisation, dé-bruitage)
                        ──▶  RawDocument

Choix d'architecture
--------------------
- **MinerU en local (in-process)** : l'OCR tourne sur la machine via le backend
  `pipeline` de MinerU (pur CPU, pas de GPU requis). Aucun document ne sort de
  la machine — c'est un point fort pour l'argument de conformité HDS.
- L'import de MinerU (dépendances lourdes : torch, modèles) est **paresseux**
  (`ocr.py::_load_mineru`) pour garder le module importable sans MinerU installé.
"""
