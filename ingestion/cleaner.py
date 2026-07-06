"""Nettoyage et normalisation du texte clinique.

Responsabilité
--------------
Assainir les blocs parsés (`parser.py`) avant chunking et produire un
`RawDocument` :
- suppression des artefacts OCR (lignes de numéro de page, form feeds) ;
- recollage des mots coupés en fin de ligne (« inflamma-\\ntion » → « inflammation ») ;
- normalisation des espaces et des lignes vides ;
- préservation des éléments cliniquement signifiants (valeurs, unités,
  posologies) — nettoyage volontairement conservateur.

Sortie
------
Un `RawDocument` dont :
- `text` est le texte nettoyé complet (concaténation des sections) ;
- `metadata["sections"]` porte la structure section par section, réutilisée par
  le chunker pour produire des chunks alignés sur les sections cliniques.
"""

from __future__ import annotations

import re

from ingestion.parser import Block, map_section
from models import RawDocument

# Ligne ne contenant qu'un numéro de page (« 3 », « Page 3 », « - 3 - »).
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?[-–]?\s*\d+\s*[-–]?\s*$", re.IGNORECASE)
# Mot coupé par un retour à la ligne : « inflamma-\ntion ».
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)[-­]\n(\w)")
# Espaces/tabulations multiples (hors retours ligne).
_MULTISPACE_RE = re.compile(r"[ \t]+")
# 3 retours à la ligne ou plus.
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise un fragment de texte issu de l'OCR (conservateur)."""
    # Uniformisation des fins de ligne et des form feeds.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    # Recollage des mots coupés en fin de ligne.
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Nettoyage ligne à ligne : retrait des lignes « numéro de page seul ».
    kept_lines = [
        _MULTISPACE_RE.sub(" ", line).rstrip()
        for line in text.split("\n")
        if not _PAGE_NUMBER_RE.match(line)
    ]
    text = "\n".join(kept_lines)

    # Compactage des lignes vides multiples et rognage global.
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_document(
    doc_id: str, source_path: str, blocks: list[Block]
) -> RawDocument:
    """Normalise les blocs parsés en un `RawDocument`.

    Args:
        doc_id: identifiant du document.
        source_path: chemin du fichier source.
        blocks: blocs structurés issus de `parser.parse_markdown`.

    Returns:
        Un `RawDocument` nettoyé, avec la structure par section dans
        `metadata["sections"]`.
    """
    sections: list[dict] = []
    text_parts: list[str] = []

    for block in blocks:
        cleaned = clean_text(block.get("text", ""))
        if not cleaned:
            continue
        title = block.get("title")
        section = map_section(title)
        sections.append(
            {
                "section": section.value,
                "title": title,
                "text": cleaned,
            }
        )
        # Le titre est conservé en tête du texte plat pour garder le contexte.
        text_parts.append(f"{title}\n{cleaned}" if title else cleaned)

    full_text = "\n\n".join(text_parts).strip()
    return RawDocument(
        doc_id=doc_id,
        source_path=source_path,
        text=full_text,
        metadata={"sections": sections},
    )
