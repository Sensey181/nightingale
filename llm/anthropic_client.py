"""Implémentation `LLMClient` sur l'API Anthropic (claude-sonnet-4-6, démo).

Détails d'implémentation (SDK officiel `anthropic`)
--------------------------------------------------
- Client : `anthropic.Anthropic(api_key=...)`. La clé provient de la config
  (`Settings.anthropic_api_key`, chargée depuis `.env`) — jamais codée en dur.
  On la passe explicitement plutôt que de se reposer sur `os.environ`, car
  pydantic-settings lit `.env` sans forcément l'exporter dans l'environnement.
- Appel : `client.messages.create(model, max_tokens, system=SYSTEM_PROMPT,
  messages=[{"role": "user", "content": build_user_prompt(...)}])`.
- Modèle : `claude-sonnet-4-6` (configurable via `Settings.llm_model`).
- Réponse : on itère `response.content` et on concatène les blocs `type ==
  "text"` (on ne suppose pas que `content[0]` est du texte).

L'import du SDK est paresseux (module importable sans `anthropic` installé).

Swap production
---------------
Interchangeable avec un client Ollama local implémentant la même interface
`LLMClient` — voir `scripts/migrate_to_local.py`.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings
from llm.base import LLMClient, LLMError
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from models import RetrievedChunk

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Client de génération basé sur l'API Anthropic."""

    def __init__(
        self, settings: Settings | None = None, max_tokens: int | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.max_tokens = max_tokens or self.settings.llm_max_tokens
        self._client = None  # anthropic.Anthropic (chargé paresseusement)

    def _ensure_client(self):
        if self._client is None:
            if not self.settings.anthropic_api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY est absente : renseignez-la dans le "
                    "fichier .env."
                )
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key
            )
        return self._client

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        """Génère la réponse via `client.messages.create`.

        Args:
            question: la question médicale.
            context: les chunks pertinents (déjà rerankés).

        Returns:
            La réponse générée, ancrée sur le contexte et citant ses sources.

        Raises:
            LLMError: clé absente ou échec de l'appel API.
        """
        client = self._ensure_client()
        try:
            response = client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": build_user_prompt(question, context),
                    }
                ],
            )
        except Exception as exc:  # erreurs SDK/API
            raise LLMError(f"Échec de l'appel à l'API Anthropic : {exc}") from exc

        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Concatène les blocs de texte de la réponse (ignore thinking, etc.)."""
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
