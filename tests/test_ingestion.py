"""Tests du module d'ingestion : parser, cleaner et flux OCR (mocké).

Le parsing et le nettoyage sont testés directement (logique pure). Le client
MinerU (local, in-process) est testé en injectant des doublures de `do_parse`
et `read_fn` — MinerU n'étant pas installé dans le python système, l'import
paresseux (`_load_mineru`) est monkeypatché.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from ingestion.cleaner import clean_document, clean_text
from ingestion.parser import map_section, parse_markdown
from models import ClinicalSection

# --------------------------------------------------------------------------- #
# parser                                                                      #
# --------------------------------------------------------------------------- #


def test_parse_markdown_decoupe_par_titres() -> None:
    md = (
        "En-tête du document\n"
        "# Chief Complaint\n"
        "Chest pain.\n"
        "## Past Medical History\n"
        "Hypertension.\n"
    )
    blocks = parse_markdown(md)

    titles = [b["title"] for b in blocks]
    assert titles == [None, "Chief Complaint", "Past Medical History"]
    assert blocks[0]["text"] == "En-tête du document"
    assert blocks[1]["text"] == "Chest pain."
    assert blocks[2]["level"] == 2


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Chief Complaint", ClinicalSection.MOTIF),
        ("Past Medical History", ClinicalSection.ANTECEDENTS),
        ("History of Present Illness", ClinicalSection.HISTOIRE),
        ("Physical Examination", ClinicalSection.EXAMEN),
        ("Pertinent Results", ClinicalSection.BIOLOGIE),
        ("Chest X-Ray", ClinicalSection.IMAGERIE),
        ("Discharge Medications", ClinicalSection.TRAITEMENT),
        ("Discharge Diagnosis", ClinicalSection.CONCLUSION),
        ("Random Heading", ClinicalSection.AUTRE),
        (None, ClinicalSection.AUTRE),
    ],
)
def test_map_section(title, expected) -> None:
    assert map_section(title) is expected


def test_map_section_priorite_antecedents_sur_histoire() -> None:
    # « Past Medical History » contient « history » : ne doit PAS tomber dans
    # HISTOIRE mais dans ANTECEDENTS.
    assert map_section("Past Medical History") is ClinicalSection.ANTECEDENTS


# --------------------------------------------------------------------------- #
# cleaner                                                                     #
# --------------------------------------------------------------------------- #


def test_clean_text_recolle_mots_coupes_et_retire_pages() -> None:
    raw = "inflamma-\ntion sévère\nPage 3\n\n\n\nsuite du texte"
    cleaned = clean_text(raw)
    assert "inflammation sévère" in cleaned
    assert "Page 3" not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_document_construit_rawdocument_avec_sections() -> None:
    blocks = [
        {"title": None, "level": 0, "text": "Discharge summary."},
        {"title": "Pertinent Results", "level": 2, "text": "Na 140 mmol/L."},
    ]
    doc = clean_document("doc1", "/data/raw/doc1.pdf", blocks)

    assert doc.doc_id == "doc1"
    sections = doc.metadata["sections"]
    assert len(sections) == 2
    assert sections[1]["section"] == ClinicalSection.BIOLOGIE.value
    assert "Na 140 mmol/L." in doc.text
    # La valeur biologique et son unité sont préservées.
    assert "140 mmol/L" in sections[1]["text"]


# --------------------------------------------------------------------------- #
# OCR (flux local mocké)                                                      #
# --------------------------------------------------------------------------- #


def test_ocr_to_markdown_flux_local(monkeypatch, tmp_path) -> None:
    from ingestion import ocr as ocr_module

    pdf = tmp_path / "cr.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    markdown = "# Chief Complaint\nChest pain."

    def fake_read_fn(path, file_suffix=None):
        assert Path(path) == pdf
        return b"pdf-bytes"

    # `do_parse` écrit le Markdown sur disque comme le vrai MinerU :
    # output_dir/{nom}/{méthode}/{nom}.md
    def fake_do_parse(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                      backend, parse_method, **kwargs):
        assert pdf_bytes_list == [b"pdf-bytes"]
        assert backend == "pipeline"
        name = pdf_file_names[0]
        out = Path(output_dir) / name / "auto"
        out.mkdir(parents=True)
        (out / f"{name}.md").write_text(markdown, encoding="utf-8")

    monkeypatch.setattr(
        ocr_module, "_load_mineru", lambda: (fake_do_parse, fake_read_fn)
    )

    client = ocr_module.MinerUClient(Settings())  # type: ignore[call-arg]
    result = client.ocr_to_markdown(pdf)
    assert result == markdown


def test_ocr_fichier_absent_leve_erreur(tmp_path) -> None:
    from ingestion.ocr import MinerUClient, MinerUError

    client = MinerUClient(Settings())  # type: ignore[call-arg]
    with pytest.raises(MinerUError, match="introuvable"):
        client.ocr_to_markdown(tmp_path / "absent.pdf")


def test_ocr_sans_markdown_leve_erreur(monkeypatch, tmp_path) -> None:
    from ingestion import ocr as ocr_module
    from ingestion.ocr import MinerUError

    pdf = tmp_path / "cr.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def fake_do_parse(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                      backend, parse_method, **kwargs):
        # MinerU n'écrit aucun .md (échec silencieux d'extraction).
        return None

    monkeypatch.setattr(
        ocr_module,
        "_load_mineru",
        lambda: (fake_do_parse, lambda path, file_suffix=None: b"x"),
    )

    client = ocr_module.MinerUClient(Settings())  # type: ignore[call-arg]
    with pytest.raises(MinerUError, match="aucun Markdown"):
        client.ocr_to_markdown(pdf)
