"""Interface abstraite du client de génération (LLM swappable).

Ce contrat permet d'interchanger le fournisseur de génération sans toucher au
reste du pipeline :
    - `AnthropicClient` (API, démo) ;
    - client Ollama local (production HDS full local).

Toute implémentation reçoit une question et un contexte (chunks retrievés
formatés) et retourne une réponse textuelle citée.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from models import RetrievedChunk


class LLMError(RuntimeError):
    """Erreur remontée par un client de génération (clé absente, échec API…)."""


class LLMClient(ABC):
    """Contrat minimal d'un client de génération pour ClinicalRAG."""

    @abstractmethod
    def generate(self, question: str, context: list[RetrievedChunk]) -> str:
        """Génère une réponse à partir de la question et du contexte retrievé.

        Args:
            question: la question médicale de l'utilisateur.
            context: les chunks pertinents issus du retrieval (déjà rerankés).

        Returns:
            La réponse générée, ancrée sur le contexte et citant ses sources.
        """
        raise NotImplementedError
