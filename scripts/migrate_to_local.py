"""Vérification « prêt pour le full local » (production HDS).

Objectif
--------
Le passage d'un déploiement de démonstration (API Anthropic pour la génération)
à un déploiement **intégralement local** ne nécessite plus de réécriture de code :
le fournisseur de génération est sélectionnable à chaud (bouton dans l'app, ou
`CLINICALRAG_LLM_PROVIDER=ollama` en config) grâce à l'interface `LLMClient` et à
la fabrique `llm/factory.py` (`AnthropicClient` ↔ `OllamaClient`).

L'OCR (MinerU) et le retrieval (embeddings, BM25, reranker, ChromaDB) tournent
déjà 100 % en local. La **seule** dépendance réseau restante en démo est l'appel
de génération à Anthropic ; basculer sur Ollama la supprime.

Ce script ne migre donc rien de destructif : il **vérifie l'état de préparation**
d'un déploiement full local et affiche un rapport actionnable :
    1. GPU CUDA disponible ? (accélère embeddings / reranker / OCR `vlm`) ;
    2. serveur Ollama joignable et modèle récupéré ? (génération locale) ;
    3. aucune clé API externe requise une fois `llm_provider == "ollama"`.

Usage :
    python -m scripts.migrate_to_local
    python scripts/migrate_to_local.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings  # noqa: E402


def check_gpu() -> bool:
    """Retourne True si un GPU CUDA est disponible (embeddings/reranker/OCR).

    Import de torch paresseux : le script reste exécutable même si torch n'est
    pas installé (on retourne alors False).
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def check_ollama(settings: Settings | None = None) -> tuple[bool, str]:
    """Vérifie que le serveur Ollama est joignable et que le modèle est présent.

    Returns:
        `(prêt, message)` : `prêt` True si le serveur répond ET que
        `settings.ollama_model` figure dans les modèles disponibles.
    """
    settings = settings or get_settings()
    try:
        import ollama
    except ImportError:
        return False, "package `ollama` non installé (`pip install ollama`)."

    client = ollama.Client(host=settings.ollama_host)
    try:
        response = client.list()
    except Exception as exc:  # serveur non lancé / injoignable
        return False, (
            f"serveur injoignable sur {settings.ollama_host} : {exc}. "
            "Lancez `ollama serve`."
        )

    models = _installed_models(response)
    wanted = settings.ollama_model
    if any(m == wanted or m.startswith(f"{wanted}:") for m in models):
        return True, f"serveur OK, modèle « {wanted} » disponible."
    return False, (
        f"serveur OK mais modèle « {wanted} » absent. "
        f"Récupérez-le : `ollama pull {wanted}`. "
        f"Modèles présents : {', '.join(models) or '(aucun)'}."
    )


def _installed_models(response) -> list[str]:
    """Extrait les noms de modèles d'une réponse `ollama.list` (dict ou objet)."""
    items = (
        response.get("models", [])
        if isinstance(response, dict)
        else getattr(response, "models", [])
    )
    names: list[str] = []
    for item in items:
        name = (
            item.get("model") or item.get("name")
            if isinstance(item, dict)
            else getattr(item, "model", None) or getattr(item, "name", None)
        )
        if name:
            names.append(name)
    return names


def readiness_report(settings: Settings | None = None) -> bool:
    """Affiche le rapport de préparation full local. Retourne True si prêt."""
    settings = settings or get_settings()

    gpu = check_gpu()
    ollama_ready, ollama_msg = check_ollama(settings)
    provider = settings.llm_provider.lower()

    print("Préparation d'un déploiement full local (génération → Ollama)\n")
    print(f"  Provider LLM courant : {provider}")
    print(f"  [{'x' if ollama_ready else ' '}] Ollama       : {ollama_msg}")
    print(
        f"  [{'x' if gpu else ' '}] GPU CUDA     : "
        + ("disponible (accélération possible)" if gpu else "absent (CPU only, OK)")
    )
    print(
        f"  [{'x' if provider == 'ollama' else ' '}] Full local   : "
        + (
            "actif — aucune donnée ne sort de la machine."
            if provider == "ollama"
            else "inactif — la génération passe encore par l'API Anthropic. "
            "Basculez via le bouton de l'app ou "
            "CLINICALRAG_LLM_PROVIDER=ollama."
        )
    )

    ready = ollama_ready
    print(
        "\n=> "
        + (
            "Prêt pour le full local."
            if ready
            else "Pas encore prêt : voir les points [ ] ci-dessus."
        )
    )
    return ready


def main() -> None:
    ready = readiness_report()
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
