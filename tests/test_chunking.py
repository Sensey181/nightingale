"""Tests du chunker sémantique.

Vérifient l'alignement sur les sections, la non-fusion entre sections, la
taille cible, le chevauchement, la préservation des valeurs cliniques et le
repli sans structure.
"""

from __future__ import annotations

import pytest

from chunking.semantic_chunker import SemanticChunker, chunk_document
from models import ClinicalSection, RawDocument


def _doc(sections: list[dict], text: str = "") -> RawDocument:
    return RawDocument(
        doc_id="doc1",
        source_path="/data/raw/doc1.pdf",
        text=text or " ".join(s["text"] for s in sections),
        metadata={"sections": sections},
    )


def test_section_courte_donne_un_chunk() -> None:
    doc = _doc(
        [{"section": ClinicalSection.BIOLOGIE.value, "title": "Labs", "text": "Na 140 mmol/L."}]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section is ClinicalSection.BIOLOGIE
    assert c.doc_id == "doc1"
    assert c.position == 0
    assert c.metadata["title"] == "Labs"
    assert "140 mmol/L" in c.text  # valeur + unité préservées


def test_sections_distinctes_ne_fusionnent_pas() -> None:
    doc = _doc(
        [
            {"section": ClinicalSection.MOTIF.value, "title": "CC", "text": "Chest pain."},
            {"section": ClinicalSection.TRAITEMENT.value, "title": "Meds", "text": "Aspirin 81 mg."},
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].section is ClinicalSection.MOTIF
    assert chunks[1].section is ClinicalSection.TRAITEMENT
    assert [c.position for c in chunks] == [0, 1]
    # Chaque chunk ne contient que le texte de sa propre section.
    assert "Aspirin" not in chunks[0].text
    assert "Chest pain" not in chunks[1].text


def test_section_longue_est_decoupee_avec_chevauchement() -> None:
    # 30 phrases distinctes → dépasse largement la cible.
    phrases = " ".join(f"Phrase numero {i} du compte rendu." for i in range(30))
    doc = _doc([{"section": ClinicalSection.HISTOIRE.value, "title": "HPI", "text": phrases}])

    chunker = SemanticChunker(target_chars=200, overlap_chars=60)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    # Tous rattachés à la même section, positions consécutives.
    assert all(c.section is ClinicalSection.HISTOIRE for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    # Taille bornée (cible + un peu de chevauchement).
    assert all(len(c.text) <= 200 + 60 for c in chunks)
    # Chevauchement : un fragment de fin du chunk i se retrouve au début du i+1.
    for a, b in zip(chunks, chunks[1:]):
        tail_word = a.text.split()[-1]
        assert tail_word in b.text


def test_repli_sans_sections() -> None:
    doc = RawDocument(
        doc_id="doc2",
        source_path="/data/raw/doc2.pdf",
        text="Texte sans structure de sections.",
        metadata={},
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].section is ClinicalSection.AUTRE


def test_section_vide_est_ignoree() -> None:
    doc = _doc(
        [
            {"section": ClinicalSection.MOTIF.value, "title": "CC", "text": "   "},
            {"section": ClinicalSection.CONCLUSION.value, "title": "Dx", "text": "Pneumonia."},
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].section is ClinicalSection.CONCLUSION


def test_parametres_invalides() -> None:
    with pytest.raises(ValueError):
        SemanticChunker(target_chars=0)
    with pytest.raises(ValueError):
        SemanticChunker(target_chars=100, overlap_chars=100)
