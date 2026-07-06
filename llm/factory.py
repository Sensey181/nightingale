"""Fabrique de client de génération : sélectionne le backend selon le provider.

Point d'injection **unique** du choix « démo (Anthropic) vs full local (Ollama) ».
Le reste du pipeline ne dépend que de l'interface `LLMClient` (`llm/base.py`) :
basculer de fournisseur ne touche qu'ici (et au provider en config / dans l'UI).
"""

from __future__ import annotations

from config import Settings, get_settings
from llm.base import LLMClient, LLMError

# Providers reconnus (valeurs de `Settings.llm_provider`).
PROVIDERS = ("anthropic", "ollama")


def build_llm(
    provider: str | None = None, settings: Settings | None = None
) -> LLMClient:
    """Instancie le client de génération correspondant au provider demandé.

    Args:
        provider: "anthropic" (démo) ou "ollama" (local). Par défaut, la valeur
            de `Settings.llm_provider`.
        settings: configuration (créée sinon).

    Returns:
        Une implémentation de `LLMClient` prête à `generate(...)`.

    Raises:
        LLMError: provider inconnu.
    """
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()

    if provider == "anthropic":
        from llm.anthropic_client import AnthropicClient

        return AnthropicClient(settings)
    if provider == "ollama":
        from llm.ollama_client import OllamaClient

        return OllamaClient(settings)

    raise LLMError(
        f"Provider LLM inconnu : {provider!r}. Attendu l'un de {PROVIDERS}."
    )
