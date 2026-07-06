"""Tests du module LLM : formatage du prompt et client Anthropic (mocké).

Le SDK `anthropic` n'est pas requis : le client est testé avec une doublure
injectée qui capture l'appel `messages.create` et renvoie une fausse réponse.
"""

from __future__ import annotations

import pytest

from config import Settings
from llm.anthropic_client import AnthropicClient
from llm.base import LLMClient, LLMError
from llm.factory import build_llm
from llm.ollama_client import OllamaClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt, format_context
from models import Chunk, ClinicalSection, RetrievedChunk


def _rc(cid: str, text: str, section=ClinicalSection.AUTRE, title=None) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id="doc",
        text=text,
        section=section,
        position=0,
        metadata={"title": title},
    )
    return RetrievedChunk(chunk=chunk, score=1.0, retrieval_source="reranked")


# --------------------------------------------------------------------------- #
# prompts                                                                     #
# --------------------------------------------------------------------------- #
def test_format_context_numerote_et_annote() -> None:
    ctx = format_context(
        [
            _rc("a", "Na 140 mmol/L.", ClinicalSection.BIOLOGIE, title="Labs"),
            _rc("b", "Pneumonia.", ClinicalSection.CONCLUSION),
        ]
    )
    assert "[source 1] (section : biologie — Labs)" in ctx
    assert "Na 140 mmol/L." in ctx
    assert "[source 2] (section : conclusion)" in ctx


def test_format_context_vide() -> None:
    assert "aucun extrait" in format_context([])


def test_build_user_prompt_contient_question_et_consigne() -> None:
    prompt = build_user_prompt("Motif ?", [_rc("a", "Chest pain.")])
    assert "Question : Motif ?" in prompt
    assert "[source N]" in prompt  # consigne de citation
    assert "Chest pain." in prompt


# --------------------------------------------------------------------------- #
# AnthropicClient                                                             #
# --------------------------------------------------------------------------- #
class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, blocks) -> None:
        self.content = blocks


class _FakeMessages:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        # Simule un bloc thinking (ignoré) + le texte de réponse.
        return _Response(
            [_Block("thinking", "réflexion"), _Block("text", "Réponse [source 1].")]
        )


class _FakeAnthropic:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_generate_appel_et_extraction_texte() -> None:
    client = AnthropicClient(Settings(anthropic_api_key="k", llm_max_tokens=512))  # type: ignore[call-arg]
    client._client = _FakeAnthropic()  # court-circuite le SDK

    chunks = [_rc("a", "Chest pain.", ClinicalSection.MOTIF)]
    answer = client.generate("Motif ?", chunks)

    # Seul le bloc texte est renvoyé (thinking ignoré).
    assert answer == "Réponse [source 1]."

    kwargs = client._client.messages.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 512
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["messages"][0]["role"] == "user"
    assert "Motif ?" in kwargs["messages"][0]["content"]
    assert "Chest pain." in kwargs["messages"][0]["content"]


def test_generate_sans_cle_leve_erreur() -> None:
    client = AnthropicClient(Settings(anthropic_api_key=""))  # type: ignore[call-arg]
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        client.generate("Motif ?", [])


def test_generate_erreur_api_encapsulee() -> None:
    class _Boom:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("500")

    client = AnthropicClient(Settings(anthropic_api_key="k"))  # type: ignore[call-arg]
    client._client = _Boom()
    with pytest.raises(LLMError, match="API Anthropic"):
        client.generate("Motif ?", [_rc("a", "x")])


# --------------------------------------------------------------------------- #
# OllamaClient                                                                #
# --------------------------------------------------------------------------- #
class _FakeOllama:
    """Doublure du client ollama : capture l'appel `chat` et renvoie un dict."""

    def __init__(self) -> None:
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return {"message": {"content": "Réponse locale [source 1]."}}


def test_ollama_generate_appel_et_extraction() -> None:
    client = OllamaClient(Settings(ollama_model="llama3.1", llm_max_tokens=256))  # type: ignore[call-arg]
    client._client = _FakeOllama()  # court-circuite le package ollama

    chunks = [_rc("a", "Chest pain.", ClinicalSection.MOTIF)]
    answer = client.generate("Motif ?", chunks)

    assert answer == "Réponse locale [source 1]."
    kwargs = client._client.kwargs
    assert kwargs["model"] == "llama3.1"
    assert kwargs["options"]["num_predict"] == 256
    # System prompt + user prompt (contexte + question) passés en messages.
    assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert kwargs["messages"][1]["role"] == "user"
    assert "Motif ?" in kwargs["messages"][1]["content"]
    assert "Chest pain." in kwargs["messages"][1]["content"]


def test_ollama_erreur_encapsulee() -> None:
    class _Boom:
        @staticmethod
        def chat(**kwargs):
            raise RuntimeError("connection refused")

    client = OllamaClient(Settings(ollama_model="llama3.1"))  # type: ignore[call-arg]
    client._client = _Boom()
    with pytest.raises(LLMError, match="Ollama"):
        client.generate("Motif ?", [_rc("a", "x")])


# --------------------------------------------------------------------------- #
# Fabrique build_llm                                                          #
# --------------------------------------------------------------------------- #
def test_build_llm_selectionne_le_provider() -> None:
    settings = Settings(anthropic_api_key="k")  # type: ignore[call-arg]
    assert isinstance(build_llm("anthropic", settings), AnthropicClient)
    assert isinstance(build_llm("ollama", settings), OllamaClient)
    # Défaut = Settings.llm_provider.
    assert isinstance(build_llm(settings=settings), LLMClient)


def test_build_llm_provider_inconnu_leve_erreur() -> None:
    with pytest.raises(LLMError, match="inconnu"):
        build_llm("gpt4")
