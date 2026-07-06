"""Tests du module de gestion du corpus (`corpus.py`).

L'OCR et l'indexation lourds (`ingest_file`, `index_documents`) sont remplacés
par des doublures ; on vérifie la cohérence entre la base vectorielle (store
factice) et la persistance JSON dans `data/processed`.
"""

from __future__ import annotations

import corpus
from config import Settings
from models import RawDocument


class _FakeStore:
    """Store factice : enregistre les appels, renvoie des compteurs fixes."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.labels: dict[str, str] = {}
        self.indexed: list = []

    def delete_document(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 3

    def set_label(self, doc_id: str, label: str) -> int:
        self.labels[doc_id] = label
        return 2


def _settings(tmp_path) -> Settings:
    return Settings(processed_dir=tmp_path, raw_dir=tmp_path)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# delete_document                                                             #
# --------------------------------------------------------------------------- #
def test_delete_document_supprime_chunks_et_json(tmp_path) -> None:
    settings = _settings(tmp_path)
    (tmp_path / "docA.json").write_text("{}", encoding="utf-8")
    store = _FakeStore()

    n = corpus.delete_document("docA", settings=settings, store=store)

    assert n == 3
    assert store.deleted == ["docA"]
    assert not (tmp_path / "docA.json").exists()


def test_delete_document_json_absent_ok(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = _FakeStore()
    # Aucun JSON présent : ne doit pas lever.
    assert corpus.delete_document("ghost", settings=settings, store=store) == 3


# --------------------------------------------------------------------------- #
# set_document_label                                                          #
# --------------------------------------------------------------------------- #
def test_set_document_label_persiste_dans_json(tmp_path) -> None:
    settings = _settings(tmp_path)
    doc = RawDocument(doc_id="docA", source_path="x", text="t", label="")
    (tmp_path / "docA.json").write_text(doc.model_dump_json(), encoding="utf-8")
    store = _FakeStore()

    n = corpus.set_document_label("docA", "cardio", settings=settings, store=store)

    assert n == 2
    assert store.labels == {"docA": "cardio"}
    reloaded = RawDocument.model_validate_json(
        (tmp_path / "docA.json").read_text(encoding="utf-8")
    )
    assert reloaded.label == "cardio"


def test_set_document_label_sans_json(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = _FakeStore()
    # Pas de JSON : met à jour le store, ne crée pas de fichier.
    assert corpus.set_document_label("docA", "x", settings=settings, store=store) == 2
    assert not (tmp_path / "docA.json").exists()


# --------------------------------------------------------------------------- #
# add_document                                                                #
# --------------------------------------------------------------------------- #
def test_add_document_ingest_label_persiste_et_indexe(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    store = _FakeStore()

    def _fake_ingest(path):
        return RawDocument(doc_id="newdoc", source_path=str(path), text="texte")

    captured: dict = {}

    def _fake_index(documents, settings=None, store=None):
        captured["documents"] = documents
        captured["store"] = store
        return 5

    monkeypatch.setattr(corpus, "ingest_file", _fake_ingest)
    monkeypatch.setattr(corpus, "index_documents", _fake_index)

    doc_id, n_chunks = corpus.add_document(
        tmp_path / "f.pdf", label="cardio", settings=settings, store=store
    )

    assert (doc_id, n_chunks) == ("newdoc", 5)
    # Le label est appliqué au document indexé…
    assert captured["documents"][0].label == "cardio"
    assert captured["store"] is store
    # …et persisté dans le RawDocument JSON.
    saved = RawDocument.model_validate_json(
        (tmp_path / "newdoc.json").read_text(encoding="utf-8")
    )
    assert saved.label == "cardio"
