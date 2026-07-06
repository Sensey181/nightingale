# Nightingale

<p align="center">
  <img src="cover_picture/nightingale.png" alt="Nightingale" width="720">
</p>

**Nightingale** — un moteur de recherche intelligent pour comptes-rendus
cliniques, qui tourne entièrement en local.

Vous déposez des lettres de sortie ou des CR médicaux (PDF, images), vous posez
une question en langage naturel, et l'outil retrouve les passages pertinents
puis rédige une réponse synthétique : toujours sourcée, sans jamais inventer.

Tout le traitement (lecture OCR des documents, recherche, indexation) se fait
sur la machine, sans qu'aucune donnée ne sorte : un choix pensé pour les
contraintes de confidentialité du secteur de la santé (HDS). La génération de la
réponse peut au choix passer par un modèle cloud (pour la démo) ou par un modèle
100 % local, activable d'un clic.

---

## Sommaire

- [Architecture](#architecture)
- [Stack technique & choix justifiés](#stack-technique--choix-justifiés)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Génération : cloud ou 100 % local](#génération--cloud-ou-100--local)
- [Conformité HDS](#conformité-hds)
- [Données](#données)

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
   PDF / lettres        │              INGESTION (local)           │
   de sortie   ───────▶ │  MinerU (OCR local) ▸ parsing ▸ nettoyage│ ──▶ RawDocument
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
                        │  ☁️ Anthropic (claude-sonnet-4-6)         │ ──▶ réponse citée
                        │  🔒 Ollama local — bascule d'un clic      │
                        └──────────────────────────────────────────┘
                                            │
                        ┌──────────────────────────────────────────┐
                        │            INTERFACE (Streamlit)         │
                        └──────────────────────────────────────────┘
```

**Principe directeur** : toute la chaîne qui touche aux **données** (OCR,
ingestion, index vectoriel, recherche) tourne **en local**, sur CPU, sans GPU
requis. Le seul maillon qui peut sortir de la machine est la **génération** de
la réponse — et encore, uniquement si l'on choisit le mode cloud pour la démo :
le mode local le ramène entièrement sur la machine.

---

## Stack technique & choix justifiés

| Étape | Techno | Pourquoi ce choix |
|-------|--------|-------------------|
| **OCR** | MinerU (local, in-process, backend `pipeline` CPU) | Extraction sur la machine : aucun document ne sort du SI dès la lecture. Bonne restitution de la structure (utile au chunking par section). |
| **Embeddings** | `multilingual-e5-small` (local, CPU) | Léger, multilingue (français clinique), tourne sur CPU. Aucune donnée envoyée à un service d'embedding externe. |
| **Vector store** | ChromaDB (persisté local) | Index entièrement sur disque, aucun appel réseau → conforme HDS côté données. Simple à opérer. |
| **Recherche hybride** | BM25 + vectoriel, fusion **RRF** | BM25 capture les termes exacts (molécules, codes, valeurs) ; le vectoriel capture le sens. RRF fusionne les *rangs* sans calibrer des scores d'échelles incomparables. |
| **Reranker** | `ms-marco-MiniLM-L6-v2` (cross-encoder, CPU) | Ré-ordonne finement les candidats fusionnés → contexte réduit et plus pertinent pour le LLM. Local, sans GPU. |
| **LLM** | Anthropic `claude-sonnet-4-6` (cloud) **ou** Ollama (local), commutable d'un clic | Interface abstraite `LLMClient` : le cloud offre une qualité de démo, le local supprime toute dépendance réseau. Le choix se fait à chaud dans l'app. |
| **Orchestration** | Pipeline custom (pas de LangChain) | Contrôle total du flux, dépendances minimales, code lisible et auditable. |
| **Interface** | Streamlit | Démo rapide : question → réponse citée + sources, gestion du corpus, ajout de documents. |

---

## Structure du dépôt

```
.
├── config.py                # configuration centralisée (pydantic-settings)
├── models.py                # schémas de données partagés (Pydantic)
├── corpus.py                # opérations « document » (ajout, suppression, label) pour l'UI
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
├── llm/                     # génération (interface swappable cloud/local)
│   ├── base.py              # interface abstraite LLMClient
│   ├── anthropic_client.py  # implémentation cloud (API Anthropic)
│   ├── ollama_client.py     # implémentation locale (Ollama)
│   ├── factory.py           # sélection du backend selon le provider
│   └── prompts.py           # prompt médical + formatage du contexte
├── evaluation/              # métriques de retrieval (MRR, NDCG) + harnais
│   ├── metrics.py
│   └── harness.py
├── app/                     # interface Streamlit de démo
│   └── streamlit_app.py
├── scripts/
│   └── migrate_to_local.py  # rapport de préparation au full local (GPU, Ollama)
├── tests/                   # tests de fumée + tests par module
├── data/                    # raw / processed / chroma (ignorés par git) ; eval versionné
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

# 3. Dépendances (retrieval, OCR MinerU local, clients LLM)
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# puis, pour le mode cloud, renseigner ANTHROPIC_API_KEY dans .env
# (l'OCR MinerU est local, sans clé)
```

**Mode 100 % local (optionnel)** — pour générer les réponses sans réseau via
Ollama :

```bash
# installer Ollama (https://ollama.com), puis :
ollama pull llama3.2:1b        # modèle léger CPU (défaut) ; un modèle plus gros
                               # (ex. llama3.1:8b) améliore l'ancrage et les citations
python scripts/migrate_to_local.py   # rapport : Ollama joignable ? modèle prêt ?
```

> ⚠️ Le fichier `.env` est ignoré par git. Ne committez jamais de clé API.

---

## Utilisation

Placer les documents sources (PDF/images) dans `data/raw/` (ou les ajouter
directement depuis l'app), puis :

```bash
# 1. Ingestion : OCR (MinerU local) → parsing → nettoyage → data/processed/*.json
python -m ingestion -v

# 2. Indexation : chunking → embeddings (e5-small) → ChromaDB
python -m indexing --reset -v

# 3a. Retrieval seul (debug) : affiche les chunks retenus
python -m retrieval "Quel est le motif d'hospitalisation ?" -v

# 3b. RAG complet en CLI : retrieval + génération + sources
python -m llm "Quel est le motif d'hospitalisation ?"

# 4. Interface de démo (recherche, corpus, ajout de documents, bascule cloud/local)
streamlit run app/streamlit_app.py

# 5. Évaluation du retrieval sur un jeu de questions annotées
python -m evaluation data/eval/questions.json --k 5 --id-field chunk
```

Sur le jeu de démo (12 questions annotées au chunk, 2 lettres de sortie) :
**MRR ≈ 0.73**, **nDCG@5 ≈ 0.78**.

---

## Génération : cloud ou 100 % local

La génération est volontairement **découplée** derrière l'interface abstraite
`llm/base.py::LLMClient`, avec deux implémentations interchangeables :

| Backend | Implémentation | Usage |
|---------|----------------|-------|
| ☁️ **Cloud** | `AnthropicClient` (`claude-sonnet-4-6`) | démo : meilleure qualité de réponse, nécessite `ANTHROPIC_API_KEY` |
| 🔒 **Local** | `OllamaClient` (Ollama, ex. `llama3.2:1b`) | full local : aucune donnée ne sort de la machine |

Le choix se fait **à chaud dans l'app** (bouton *Modèle de génération*), sans
rien changer au reste du pipeline — c'est `llm/factory.py` qui instancie le bon
client selon `CLINICALRAG_LLM_PROVIDER`. Le retrieval étant déjà local, choisir
le backend Ollama rend le pipeline **intégralement on-premise**.

Le script [`scripts/migrate_to_local.py`](scripts/migrate_to_local.py) fournit un
**rapport de préparation** au full local : GPU disponible ? serveur Ollama
joignable ? modèle récupéré ? provider actif ?

---

## Conformité HDS

Le projet est conçu pour respecter les exigences **HDS** côté données :

- **OCR 100 % local** : MinerU tourne sur la machine (backend `pipeline`, pur
  CPU). Aucun document ne sort du SI dès l'étape d'extraction.
- **Retrieval 100 % local** : parsing, index vectoriel (ChromaDB), embeddings
  et reranking tournent sur la machine. Aucune donnée indexée ne sort du SI.
- **Génération commutable** : le mode cloud (API Anthropic) est le **seul**
  maillon externe, acceptable en démo car le corpus est fictif/anonymisé. Le
  mode local (Ollama) le supprime.

En clair : **en mode full local, aucune donnée patient ne quitte le SI** — le
pipeline est intégralement on-premise.

---

## Données

**Démo** : lettres de sortie **synthétiques** (fictives, générées pour la
démonstration) — aucun patient réel. Le pipeline est conçu pour des
comptes-rendus cliniques anonymisés de type **MIMIC-IV**.

Les répertoires de données (`data/raw`, `data/processed`, `data/chroma`) sont
ignorés par git — aucun compte-rendu n'est versionné. Seul le jeu d'évaluation
(`data/eval/questions.json`, questions synthétiques) est suivi.
