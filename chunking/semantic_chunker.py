"""Chunking sémantique par section clinique.

Responsabilité
--------------
Découper un `RawDocument` en `Chunk` alignés sur les sections cliniques, avec
une taille cible et un chevauchement configurables.

Entrée
------
Le chunker consomme `RawDocument.metadata["sections"]` (produit par le cleaner) :
une liste de `{"section": <valeur ClinicalSection>, "title": str|None,
"text": str}`. À défaut de structure (métadonnée absente), tout le
`document.text` est traité comme une unique section `AUTRE`.

Stratégie
---------
1. Ne jamais fusionner deux sections différentes dans un même chunk.
2. Découper le texte de chaque section en « unités » naturelles (paragraphes,
   puis phrases si un paragraphe dépasse la taille cible).
3. Empaqueter gloutonnement ces unités jusqu'à `target_chars`, avec un
   chevauchement de `overlap_chars` entre chunks consécutifs (report de la fin
   du chunk précédent, aligné sur une frontière de mot) pour préserver le
   contexte.

Chaque `Chunk` conserve `doc_id`, `section`, `position` (index ordinal dans le
document) et le titre de section en métadonnée, pour la traçabilité et le
filtrage éventuel au retrieval.
"""

from __future__ import annotations

import re

from models import Chunk, ClinicalSection, RawDocument

# Frontière de phrase : ponctuation forte suivie d'un espace. Le « \s » évite
# de couper sur les décimales cliniques (« 5.2 mmol/L » n'a pas d'espace après
# le point).
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
# Séparateur de paragraphes : au moins une ligne vide.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


class SemanticChunker:
    """Segmenteur sémantique paramétrable (taille cible, chevauchement)."""

    def __init__(self, target_chars: int = 1000, overlap_chars: int = 150) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars doit être strictement positif.")
        if not 0 <= overlap_chars < target_chars:
            raise ValueError("overlap_chars doit vérifier 0 <= overlap < target.")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    # ------------------------------------------------------------------ #
    # API publique                                                       #
    # ------------------------------------------------------------------ #
    def chunk(self, document: RawDocument) -> list[Chunk]:
        """Découpe un document en chunks alignés sur les sections cliniques.

        Args:
            document: le `RawDocument` nettoyé (avec `metadata["sections"]`).

        Returns:
            La liste ordonnée des `Chunk`.
        """
        chunks: list[Chunk] = []
        position = 0
        for section_enum, title, text in self._iter_sections(document):
            for piece in self._chunk_section(text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::{position:04d}",
                        doc_id=document.doc_id,
                        text=piece,
                        section=section_enum,
                        position=position,
                        metadata={"title": title, "label": document.label},
                    )
                )
                position += 1
        return chunks

    # ------------------------------------------------------------------ #
    # Étapes internes                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iter_sections(
        document: RawDocument,
    ) -> list[tuple[ClinicalSection, str | None, str]]:
        """Normalise les sections du document en (section, titre, texte)."""
        raw_sections = document.metadata.get("sections")
        if not raw_sections:
            # Aucune structure : le document entier est une section AUTRE.
            text = document.text.strip()
            return [(ClinicalSection.AUTRE, None, text)] if text else []

        sections: list[tuple[ClinicalSection, str | None, str]] = []
        for entry in raw_sections:
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            try:
                section_enum = ClinicalSection(entry.get("section"))
            except ValueError:
                section_enum = ClinicalSection.AUTRE
            sections.append((section_enum, entry.get("title"), text))
        return sections

    def _chunk_section(self, text: str) -> list[str]:
        """Découpe le texte d'une section en chunks avec chevauchement."""
        units = self._split_into_units(text)
        if not units:
            return []

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current} {unit}".strip() if current else unit
            if not current or len(candidate) <= self.target_chars:
                current = candidate
            else:
                chunks.append(current)
                tail = self._overlap_tail(current)
                current = f"{tail} {unit}".strip() if tail else unit
        if current:
            chunks.append(current)
        return chunks

    def _split_into_units(self, text: str) -> list[str]:
        """Segmente un texte en unités <= `target_chars` (paragraphes/phrases)."""
        units: list[str] = []
        for paragraph in _PARAGRAPH_RE.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self.target_chars:
                units.append(paragraph)
                continue
            # Paragraphe trop long : découpe en phrases.
            for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) <= self.target_chars:
                    units.append(sentence)
                else:
                    units.extend(self._hard_wrap(sentence))
        return units

    def _hard_wrap(self, text: str) -> list[str]:
        """Découpe brutalement (par mots) un fragment plus long que la cible."""
        words = text.split()
        pieces: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if not current or len(candidate) <= self.target_chars:
                current = candidate
            else:
                pieces.append(current)
                current = word
        if current:
            pieces.append(current)
        return pieces

    def _overlap_tail(self, text: str) -> str:
        """Extrait la fin de `text` (~`overlap_chars`), alignée sur un mot."""
        if self.overlap_chars <= 0 or len(text) <= self.overlap_chars:
            return ""
        tail = text[-self.overlap_chars :]
        # Repart au début du mot suivant pour ne pas couper un mot en deux.
        space = tail.find(" ")
        return tail[space + 1 :] if space != -1 else tail


def chunk_document(
    document: RawDocument,
    target_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Raccourci fonctionnel : instancie un `SemanticChunker` et découpe."""
    return SemanticChunker(target_chars, overlap_chars).chunk(document)
