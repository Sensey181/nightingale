"""Configuration centralisée de ClinicalRAG.

Toute la configuration du pipeline est concentrée ici et alimentée par les
variables d'environnement (fichier `.env`, voir `.env.example`). Aucun secret
n'est codé en dur : la clé API Anthropic (génération, démo) est lue depuis
l'environnement. L'OCR (MinerU) tourne en local et ne requiert aucune clé.

Architecture :
    - Un unique objet `Settings` (pydantic-settings) est exposé via `get_settings()`.
    - Les modules du pipeline (ingestion, chunking, indexing, retrieval, llm)
      importent cet objet plutôt que de lire `os.environ` directement.
    - Ce point de centralisation facilite le swap « démo → full local »
      documenté dans `scripts/migrate_to_local.py` : basculer le LLM vers
      Ollama ou MinerU vers l'offline ne touche qu'à cette configuration.

Conformité HDS :
    - Les chemins de données (`raw`, `processed`, `chroma`) restent locaux, et
      l'OCR (MinerU) tourne désormais 100 % en local.
    - Seule `ANTHROPIC_API_KEY` (génération, démo) sort de la machine. En
      production, le LLM local (Ollama) supprime cette dernière dépendance et
      rend le pipeline entièrement local.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres du pipeline, chargés depuis l'environnement / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLINICALRAG_",
        extra="ignore",
        populate_by_name=True,  # autorise l'init par nom de champ ET par alias
    )

    # --- Génération (LLM) ---
    # `llm_provider` choisit le backend de génération, interchangeable à chaud
    # (bouton dans l'app) sans toucher au reste du pipeline :
    #   - "anthropic" : API Anthropic (démo, une dépendance réseau) ;
    #   - "ollama"    : LLM local via Ollama (full local, conformité HDS).
    llm_provider: str = "anthropic"
    # `anthropic_api_key` n'a PAS le préfixe CLINICALRAG_ : on lit la variable
    # standard ANTHROPIC_API_KEY que le SDK Anthropic reconnaît nativement.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048  # réponse RAG : synthèse courte et citée

    # --- Génération locale (Ollama) ---
    # Modèle et serveur Ollama utilisés quand `llm_provider == "ollama"`. Le
    # modèle doit avoir été récupéré au préalable (`ollama pull <modèle>`).
    # Défaut `llama3.2:1b` : léger, tourne sur CPU sans GPU (démo full local).
    # Un modèle plus gros (ex. `llama3.1:8b`) améliore nettement l'ancrage et la
    # citation des sources, au prix de la latence sur CPU.
    ollama_model: str = "llama3.2:1b"
    ollama_host: str = "http://localhost:11434"

    # --- OCR (MinerU local, in-process) ---
    # MinerU tourne en local, sans réseau. `pipeline` = backend pur CPU (pas de
    # GPU requis) ; `vlm`/`hybrid` nécessitent un GPU.
    mineru_backend: str = "pipeline"
    mineru_parse_method: str = "auto"       # "auto" | "txt" | "ocr"
    # `en` : corpus de démo MIMIC-IV (anglais). Indice de langue du pipeline OCR.
    mineru_language: str = "en"
    mineru_enable_formula: bool = False     # formules inutiles en clinique
    mineru_enable_table: bool = True        # tableaux de biologie : à conserver
    # Miroir de téléchargement des modèles (ex. "modelscope" si le hub par
    # défaut est inaccessible). Vide = hub par défaut de MinerU.
    mineru_model_source: str = ""

    # --- Modèles locaux (CPU) ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # --- Chemins ---
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    chroma_dir: Path = Path("./data/chroma")
    chroma_collection: str = "clinical_reports"

    # --- Retrieval ---
    top_k_vector: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    top_n_rerank: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne l'instance unique (mise en cache) de la configuration."""
    return Settings()  # type: ignore[call-arg]
