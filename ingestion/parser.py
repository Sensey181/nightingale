"""Parsing du Markdown OCR en blocs structurés + détection de section.

Responsabilité
--------------
Transformer le Markdown renvoyé par MinerU (`ocr.py`) en une liste de blocs
exposant la hiérarchie du document (titres de sections + contenu). Chaque bloc
se voit rattacher une `ClinicalSection` par correspondance de mots-clés sur son
titre. Ces blocs alimentent le nettoyage (`cleaner.py`) puis le chunking
sémantique par section.

Détection de section
--------------------
Les comptes-rendus (MIMIC-IV, anglais ; CR français) suivent des intitulés
récurrents. `map_section()` associe un titre à une `ClinicalSection` via une
table de mots-clés ordonnée (du plus spécifique au plus générique) pour éviter
les collisions — p. ex. « Past Medical History » (antécédents) ne doit pas
tomber dans « History of Present Illness » (histoire de la maladie).
"""

from __future__ import annotations

import re

from models import ClinicalSection

# Un bloc = un titre de section détecté + son contenu.
# Format : {"title": str | None, "level": int, "text": str}
Block = dict

# Ligne de titre Markdown : `# ...`, `## ...`, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Table de correspondance titre → section clinique.
# ORDRE IMPORTANT : les entrées les plus spécifiques d'abord. `map_section`
# renvoie la première section dont un mot-clé apparaît dans le titre.
_SECTION_KEYWORDS: list[tuple[ClinicalSection, tuple[str, ...]]] = [
    (
        ClinicalSection.MOTIF,
        ("chief complaint", "reason for admission", "reason for visit", "motif"),
    ),
    (
        ClinicalSection.ANTECEDENTS,
        (
            "past medical history",
            "past surgical history",
            "medical history",
            "pmh",
            "antecedent",
            "antécédent",
            "antecedents",
        ),
    ),
    (
        ClinicalSection.HISTOIRE,
        (
            "history of present illness",
            "present illness",
            "hpi",
            "histoire de la maladie",
            "histoire de la maladie actuelle",
        ),
    ),
    (
        ClinicalSection.EXAMEN,
        (
            "physical exam",
            "physical examination",
            "examination",
            "exam on admission",
            "examen clinique",
            "examen physique",
        ),
    ),
    (
        ClinicalSection.BIOLOGIE,
        (
            "laboratory",
            "labs",
            "lab results",
            "pertinent results",
            "biolog",
            "bilan biologique",
        ),
    ),
    (
        ClinicalSection.IMAGERIE,
        (
            "imaging",
            "radiolog",
            "chest x-ray",
            "ct scan",
            "mri",
            "ultrasound",
            "imagerie",
            "échographie",
            "scanner",
        ),
    ),
    (
        ClinicalSection.TRAITEMENT,
        (
            "discharge medications",
            "medications on admission",
            "medications",
            "treatment",
            "traitement",
            "prescription",
            "ordonnance",
        ),
    ),
    (
        ClinicalSection.CONCLUSION,
        (
            "discharge diagnosis",
            "final diagnosis",
            "impression",
            "assessment and plan",
            "assessment",
            "discharge condition",
            "discharge instructions",
            "brief hospital course",
            "conclusion",
            "summary",
            "synthèse",
            "conduite à tenir",
        ),
    ),
]


def map_section(title: str | None) -> ClinicalSection:
    """Associe un titre de section à une `ClinicalSection`.

    Args:
        title: le texte du titre Markdown (ou None si aucun titre).

    Returns:
        La `ClinicalSection` correspondante, ou `ClinicalSection.AUTRE` si
        aucun mot-clé ne correspond.
    """
    if not title:
        return ClinicalSection.AUTRE
    normalized = title.strip().lower()
    for section, keywords in _SECTION_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return section
    return ClinicalSection.AUTRE


def parse_markdown(markdown: str) -> list[Block]:
    """Découpe le Markdown en blocs structurés (titre, niveau, contenu).

    Chaque titre Markdown (`#` … `######`) ouvre un nouveau bloc ; le texte qui
    suit, jusqu'au titre suivant, en constitue le contenu. Le texte précédant le
    premier titre (préambule / en-tête) forme un bloc initial sans titre.

    Args:
        markdown: sortie brute de MinerU.

    Returns:
        Liste ordonnée de blocs `{"title", "level", "text"}`. Les blocs vides
        (ni titre ni contenu) sont écartés.
    """
    blocks: list[Block] = []
    current: Block = {"title": None, "level": 0, "text_lines": []}

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            _flush(blocks, current)
            current = {
                "title": heading.group(2).strip(),
                "level": len(heading.group(1)),
                "text_lines": [],
            }
        else:
            current["text_lines"].append(line)

    _flush(blocks, current)
    return blocks


def _flush(blocks: list[Block], block: Block) -> None:
    """Finalise un bloc en cours et l'ajoute s'il porte du contenu utile."""
    text = "\n".join(block["text_lines"]).strip()
    title = block["title"]
    if not text and not title:
        return
    blocks.append({"title": title, "level": block["level"], "text": text})
