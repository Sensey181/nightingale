"""Tests de fumée du scaffold.

Vérifient uniquement que la structure est cohérente : les modules s'importent,
la configuration se charge, et les schémas de données se construisent. Les
tests fonctionnels par module viendront avec chaque implémentation.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "config",
    "models",
    "ingestion.ocr",
    "ingestion.parser",
    "ingestion.cleaner",
    "chunking.semantic_chunker",
    "indexing.embeddings",
    "indexing.vector_store",
    "retrieval.bm25",
    "retrieval.vector_search",
    "retrieval.fusion",
    "retrieval.reranker",
    "retrieval.pipeline",
    "llm.base",
    "llm.prompts",
    "llm.anthropic_client",
    "evaluation.metrics",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importe(module_name: str) -> None:
    """Chaque module du pipeline doit s'importer sans erreur."""
    importlib.import_module(module_name)


def test_config_se_charge() -> None:
    """La configuration se charge avec ses valeurs par défaut."""
    from config import get_settings

    settings = get_settings()
    assert settings.llm_model == "claude-sonnet-4-6"
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.rrf_k == 60


def test_schemas_se_construisent() -> None:
    """Les schémas partagés se construisent et s'imbriquent correctement."""
    from models import Chunk, ClinicalSection, RetrievedChunk

    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Kaliémie à 5.2 mmol/L.",
        section=ClinicalSection.BIOLOGIE,
        position=0,
    )
    result = RetrievedChunk(chunk=chunk, score=0.87, retrieval_source="reranked")
    assert result.chunk.section is ClinicalSection.BIOLOGIE
    assert result.score == pytest.approx(0.87)
