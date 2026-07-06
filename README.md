# ClinicalRAG

Pipeline **RAG local** sur comptes-rendus cliniques et lettres de sortie
médicales. Projet portfolio : ingestion, chunking, indexing et retrieval
tournent en local sur une machine sans GPU ; le compute lourd (OCR, génération)
est déchargé vers des API en ligne pour la démo, et le tout est conçu pour
basculer en **full local** en production.

---

## Sommaire

- [Architecture](#architecture)
- [Stack technique & choix justifiés](#stack-technique--choix-justifiés)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Conformité HDS](#conformité-hds)
- [Swap LLM local pour la production](#swap-llm-local-pour-la-production)
- [Données](#données)

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
   PDF / lettres        │              INGESTION                   │
   de sortie   ───────▶ │  MinerU (local) ▸ parsing ▸ nettoyage    │ ──▶ RawDocument
                        └──────────────────────────────────────────┘
                                            │
                        ┌──────────────────────────────────────────┐
                        │              CHUNKING                    │
                        │  segmentation sémantique par section     │ ──▶ Chunks
                        │  (motif, antécédents, biologie, …)       │
                        └──────────────────────────────────────────┘
                                            │
                        ┌──────────────────────────────────────────┐
                        │              INDEXING (local)            │
                        │  e5-small (CPU) ▸ ChromaDB (persisté)    │
                        └──────────────────────────────────────────┘
                                            │
   question   ───────▶  ┌──────────────────────────────────────────┐
                        │             RETRIEVAL (local)            │
                        │  BM25 + vectoriel ▸ RRF ▸ reranker CPU   │ ──▶ top-N chunks
                        └──────────────────────────────────────────┘
                                            │
                        ┌──────────────────────────────────────────┐
                        │             GÉNÉRATION                    │
                        │  API Anthropic (claude-sonnet-4-6)       │ ──▶ réponse citée
                        │  ◂ swappable → Ollama local (prod)       │
                        └──────────────────────────────────────────┘
                                            │
                        ┌──────────────────────────────────────────┐
                        │            INTERFACE (Streamlit)         │
                        └──────────────────────────────────────────┘
```

**Principe directeur** : tout ce qui touche aux **données** (ingestion parsée,
index vectoriel, recherche) reste **local** ; seuls l'OCR (sur documents
publics anonymisés) et la génération (démo) passent par des API externes, et
ces deux points sont conçus pour être ramenés en local (cf. conformité HDS).

---

## Stack technique & choix justifiés

| Étape | Techno | Pourquoi ce choix |
|-------|--------|-------------------|
| **OCR** | MinerU (mode **online**) | Poste de calcul le plus lourd → déchargé pour tenir la contrainte « sans GPU ». Bonne restitution de la structure (utile au chunking par section). Démo sur documents publics anonymisés. |
| **Embeddings** | `multilingual-e5-small` (local, CPU) | Léger, multilingue (français clinique), tourne sur CPU. Aucune donnée envoyée à un service d'embedding externe. |
| **Vector store** | ChromaDB (persisté local) | Index entièrement sur disque, aucun appel réseau → conforme HDS côté données. Simple à opérer. |
| **Recherche hybride** | BM25 + vectoriel, fusion **RRF** | BM25 capture les termes exacts (molécules, codes, valeurs) ; le vectoriel capture le sens. RRF fusionne les *rangs* sans calibrer des scores d'échelles incomparables. |
| **Reranker** | `ms-marco-MiniLM-L6-v2` (cross-encoder, CPU) | Ré-ordonne finement les candidats fusionnés → contexte réduit et plus pertinent pour le LLM. Local, sans GPU. |
| **LLM** | API Anthropic `claude-sonnet-4-6` (démo), **swappable Ollama** | Décharge la génération pour la démo ; interface abstraite `LLMClient` pour basculer en local sans toucher au pipeline. |
| **Orchestration** | Pipeline custom (pas de LangChain) | Contrôle total du flux, dépendances minimales, code lisible et auditable. |
| **Interface** | Streamlit | Démo rapide question → réponse citée + sources. |

---

## Structure du dépôt

```
.
├── config.py                # configuration centralisée (pydantic-settings)
├── models.py                # schémas de données partagés (Pydantic)
├── ingestion/               # OCR (MinerU local), parsing, nettoyage
│   ├── ocr.py
│   ├── parser.py
│   └── cleaner.py
├── chunking/                # segmentation sémantique par section clinique
│   └── semantic_chunker.py
├── indexing/                # embeddings (e5-small) + ChromaDB
│   ├── embeddings.py
│   └── vector_store.py
├── retrieval/               # hybride BM25+vectoriel, RRF, reranking
│   ├── bm25.py
│   ├── vector_search.py
│   ├── fusion.py            # Reciprocal Rank Fusion
│   ├── reranker.py
│   └── pipeline.py          # orchestration du retrieval
├── llm/                     # client de génération (interface swappable)
│   ├── base.py              # interface abstraite LLMClient
│   ├── anthropic_client.py  # implémentation API Anthropic (démo)
│   └── prompts.py           # prompt médical + formatage du contexte
├── evaluation/              # métriques de retrieval (MRR, NDCG)
│   └── metrics.py
├── app/                     # interface Streamlit de démo
│   └── streamlit_app.py
├── scripts/
│   └── migrate_to_local.py  # bascule démo → full local (Ollama, OCR offline, GPU)
├── tests/                   # tests de fumée + tests par module
├── data/                    # raw / processed / chroma (ignorés par git)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

```bash
# 1. Environnement virtuel
python -m venv .venv && source .venv/bin/activate

# 2. Torch CPU (machine sans GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Dépendances
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# puis renseigner ANTHROPIC_API_KEY dans .env (l'OCR MinerU est local, sans clé)
```

> ⚠️ Le fichier `.env` est ignoré par git. Ne committez jamais de clé API.

---

## Utilisation

Placer les documents sources (PDF/images) dans `data/raw/`, puis :

```bash
# 1. Ingestion : OCR (MinerU local) → parsing → nettoyage → data/processed/*.json
python -m ingestion -v

# 2. Indexation : chunking → embeddings (e5-small) → ChromaDB
python -m indexing --reset -v

# 3a. Retrieval seul (debug) : affiche les chunks retenus
python -m retrieval "Quel est le motif d'hospitalisation ?" -v

# 3b. RAG complet en CLI : retrieval + génération + sources
python -m llm "Quel est le motif d'hospitalisation ?"

# 4. Interface de démo (retrieval + génération citée)
streamlit run app/streamlit_app.py

# 5. Évaluation du retrieval sur un jeu de questions annotées
python -m evaluation data/eval/questions.json --k 5
```

Étapes intermédiaires utiles au réglage :

```bash
python -m chunking            # statistiques de chunking (sans indexer)
```

---

## Conformité HDS

Le projet est documenté comme **conforme aux exigences HDS pour la partie
données** :

- **OCR 100 % local** : MinerU tourne sur la machine (backend `pipeline`, pur
  CPU). Aucun document ne sort du SI dès l'étape d'extraction.
- **Retrieval 100 % local** : parsing, index vectoriel (ChromaDB), embeddings
  et reranking tournent sur la machine. Aucune donnée indexée ne sort du SI.
- **Génération (API Anthropic)** : **seul** maillon externe en démo, acceptable
  car le corpus est public/anonymisé. En production sur données patients
  réelles, la génération est ramenée en local (Ollama).

En clair : **en production full local, aucune donnée patient ne quitte le SI** —
le pipeline devient intégralement on-premise.

---

## Swap LLM local pour la production

La génération est volontairement **découplée** derrière l'interface abstraite
`llm/base.py::LLMClient`. Passer de la démo à la production consiste à injecter
une autre implémentation, sans toucher au reste du pipeline :

| Composant | Démo | Production (full local) |
|-----------|------|--------------------------|
| LLM | `AnthropicClient` (API, `claude-sonnet-4-6`) | `OllamaClient` (local) |
| OCR | MinerU **local** (backend `pipeline`, CPU) | idem, backend `vlm`/`hybrid` si GPU |
| GPU | non requis (CPU) | activé si disponible (embeddings/reranker/OCR) |

La checklist et l'outillage de bascule sont documentés dans
[`scripts/migrate_to_local.py`](scripts/migrate_to_local.py). Comme le retrieval
est déjà local, ce swap suffit à rendre le pipeline **entièrement on-premise**.

---

## Données

Jeu d'entraînement/test : **MIMIC-IV** (comptes-rendus cliniques anonymisés).
Les répertoires `data/` sont ignorés par git — aucun compte-rendu, même
anonymisé, n'est versionné.
