"""Schémas de données partagés du pipeline ClinicalRAG.

Ces modèles Pydantic constituent le contrat de données qui circule entre les
étapes du pipeline :

    ingestion  ──▶  RawDocument         (texte OCR nettoyé + métadonnées)
    chunking   ──▶  Chunk               (segment sémantique + section clinique)
    indexing   ──▶  (Chunk persisté dans ChromaDB avec son embedding)
    retrieval  ──▶  RetrievedChunk      (Chunk + score de pertinence)

Centraliser les schémas ici évite les dictionnaires ad hoc et garantit que
chaque module consomme/produit exactement les mêmes structures.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClinicalSection(str, Enum):
    """Sections typiques d'un compte-rendu clinique / lettre de sortie.

    Utilisées par le chunker sémantique pour rattacher chaque segment à sa
    section d'origine, ce qui améliore la pertinence du retrieval (ex. filtrer
    ou pondérer les segments « biologie » pour une question sur un bilan).
    """

    MOTIF = "motif_hospitalisation"
    ANTECEDENTS = "antecedents"
    HISTOIRE = "histoire_maladie"
    EXAMEN = "examen_clinique"
    BIOLOGIE = "biologie"
    IMAGERIE = "imagerie"
    TRAITEMENT = "traitement"
    CONCLUSION = "conclusion"
    AUTRE = "autre"


class RawDocument(BaseModel):
    """Document clinique après OCR + parsing + nettoyage (sortie d'ingestion)."""

    doc_id: str = Field(..., description="Identifiant unique du document")
    source_path: str = Field(..., description="Chemin du fichier source")
    text: str = Field(..., description="Texte nettoyé, prêt pour le chunking")
    label: str = Field(
        default="",
        description="Étiquette libre au niveau document (cohorte, service…), "
        "propagée à chaque chunk pour filtrer la recherche.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Métadonnées (source MIMIC-IV, type de CR, etc.)",
    )


class Chunk(BaseModel):
    """Segment sémantique d'un document (sortie de chunking / unité d'indexation)."""

    chunk_id: str = Field(..., description="Identifiant unique du chunk")
    doc_id: str = Field(..., description="Document parent")
    text: str = Field(..., description="Contenu textuel du segment")
    section: ClinicalSection = Field(
        default=ClinicalSection.AUTRE,
        description="Section clinique de rattachement",
    )
    position: int = Field(..., description="Index ordinal du chunk dans le document")
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Chunk retourné par le retrieval, accompagné de son score de pertinence."""

    chunk: Chunk
    score: float = Field(..., description="Score de pertinence (post-fusion/rerank)")
    retrieval_source: str = Field(
        default="hybrid",
        description="Origine : 'bm25', 'vector', 'rrf' ou 'reranked'",
    )
