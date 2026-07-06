"""Implémentation `LLMClient` sur Ollama (LLM local, full on-premise).

Rôle
----
Alternative locale à `AnthropicClient` pour la génération : aucun contenu ne
quitte la machine. Le retrieval étant déjà 100 % local, choisir ce client rend
le pipeline entièrement on-premise (argument conformité HDS sur données
patients réelles).

Détails d'implémentation (package `ollama`)
------------------------------------------
- Client : `ollama.Client(host=...)` pointant sur le serveur local
  (`Settings.ollama_host`, défaut http://localhost:11434).
- Appel : `client.chat(model, messages=[system, user], options=...)`. Le
  SYSTEM_PROMPT et le formatage du contexte (`llm/prompts.py`) sont réutilisés
  tels quels, comme pour le client Anthropic — seul le transport change.
- Modèle : `Settings.ollama_model` (doit avoir été récupéré via
  `ollama pull <modèle>`).
- `options={"num_predict": max_tokens}` borne la longueur de réponse.

L'import du package est paresseux (module importable sans `ollama` installé),
et toute erreur (serveur injoignable, modèle absent) est encapsulée en
`LLMError` avec un message actionnable.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings
from llm.base import LLMClient, LLMError
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from models import RetrievedChunk

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Client de génération basé sur un serveur Ollama local."""

    def __init__(
        self, settings: Settings | None = None, max_tokens: int | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.max_tokens = max_tokens or self.settings.llm_max_tokens
        self._client = None  # ollama.Client (chargé paresseusement)

    def _ensure_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:  # package non installé
                raise LLMError(
                    "Le package `ollama` n'est pas installé : "
                    "`pip install ollama`."
                ) from exc
            self._client = ollama.Client(host=self.settings.ollama_host)
        return self._client

    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        """Génère la réponse via `client.chat` sur le modèle local.

        Args:
            question: la question médicale.
            context: les chunks pertinents (déjà rerankés).

        Returns:
            La réponse générée, ancrée sur le contexte et citant ses sources.

        Raises:
            LLMError: package absent, serveur injoignable ou modèle indisponible.
        """
        client = self._ensure_client()
        try:
            response = client.chat(
                model=self.settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(question, context),
                    },
                ],
                options={"num_predict": self.max_tokens},
            )
        except Exception as exc:  # serveur down, modèle absent, etc.
            raise LLMError(
                f"Échec de l'appel à Ollama ({self.settings.ollama_host}, "
                f"modèle « {self.settings.ollama_model} ») : {exc}. Vérifiez que "
                "`ollama serve` tourne et que le modèle a été récupéré "
                f"(`ollama pull {self.settings.ollama_model}`)."
            ) from exc

        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Extrait le texte de la réponse `chat` (dict ou objet du SDK)."""
        message = (
            response.get("message")
            if isinstance(response, dict)
            else getattr(response, "message", None)
        )
        if message is None:
            return ""
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        return (content or "").strip()
