"""Tests de l'indexing : embeddings (préfixes e5) et VectorStore (mocké).

Aucune dépendance lourde (torch, chromadb) n'est requise : le modèle
d'embedding et la collection ChromaDB sont remplacés par des doublures en
mémoire, pour tester la logique de préfixage, de (dé)sérialisation et de
scoring.
"""

from __future__ import annotations

import pytest

from config import Settings
from indexing.embeddings import EmbeddingModel
from indexing.vector_store import VectorStore
from models import Chunk, ClinicalSection


# --------------------------------------------------------------------------- #
# Doublures                                                                   #
# --------------------------------------------------------------------------- #
class _Arr:
    """Imite un ndarray : porte `.tolist()`."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeModel:
    """Modèle d'embedding factice qui enregistre ses appels."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if isinstance(texts, str):
            return _Arr([0.1, 0.2, 0.3])
        return _Arr([[float(i)] * 3 for i in range(len(texts))])


class _FakeCollection:
    """Collection ChromaDB factice (upsert/get/query/count) en mémoire."""

    def __init__(self):
        self.ids, self.documents, self.embeddings, self.metadatas = [], [], [], []

    def upsert(self, ids, documents, embeddings, metadatas):
        for i, cid in enumerate(ids):
            if cid in self.ids:
                j = self.ids.index(cid)
                self.documents[j], self.embeddings[j], self.metadatas[j] = (
                    documents[i],
                    embeddings[i],
                    metadatas[i],
                )
            else:
                self.ids.append(cid)
                self.documents.append(documents[i])
                self.embeddings.append(embeddings[i])
                self.metadatas.append(metadatas[i])

    def count(self):
        return len(self.ids)

    def _indices(self, where=None):
        """Indices des entrées correspondant au filtre `where` (égalité simple)."""
        return [
            i
            for i, meta in enumerate(self.metadatas)
            if where is None or all(meta.get(k) == v for k, v in where.items())
        ]

    def get(self, ids=None, where=None, include=None):
        idx = self._indices(where)
        return {
            "ids": [self.ids[i] for i in idx],
            "documents": [self.documents[i] for i in idx],
            "metadatas": [self.metadatas[i] for i in idx],
        }

    def delete(self, ids=None, where=None):
        targets = set(ids or [])
        keep = [i for i, cid in enumerate(self.ids) if cid not in targets]
        self.ids = [self.ids[i] for i in keep]
        self.documents = [self.documents[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]

    def update(self, ids, embeddings=None, documents=None, metadatas=None):
        for k, cid in enumerate(ids):
            j = self.ids.index(cid)
            if metadatas is not None:
                self.metadatas[j] = metadatas[k]

    def query(self, query_embeddings, n_results, where=None, include=None):
        idx = self._indices(where)[:n_results]
        return {
            "ids": [[self.ids[i] for i in idx]],
            "documents": [[self.documents[i] for i in idx]],
            "metadatas": [[self.metadatas[i] for i in idx]],
            "distances": [[0.1 * (k + 1) for k in range(len(idx))]],
        }


def _chunk(
    i: int,
    section: ClinicalSection,
    title=None,
    doc_id: str = "doc",
    label: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{i:04d}",
        doc_id=doc_id,
        text=f"contenu {i}",
        section=section,
        position=i,
        metadata={"title": title, "label": label},
    )


# --------------------------------------------------------------------------- #
# embeddings                                                                  #
# --------------------------------------------------------------------------- #
def test_embed_passages_prefixe_et_normalise() -> None:
    em = EmbeddingModel(Settings())  # type: ignore[call-arg]
    em._model = _FakeModel()
    out = em.embed_passages(["a", "b"])

    texts, kwargs = em._model.calls[0]
    assert texts == ["passage: a", "passage: b"]
    assert kwargs["normalize_embeddings"] is True
    assert out == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_embed_query_prefixe() -> None:
    em = EmbeddingModel(Settings())  # type: ignore[call-arg]
    em._model = _FakeModel()
    out = em.embed_query("douleur thoracique")

    texts, _ = em._model.calls[0]
    assert texts == "query: douleur thoracique"
    assert out == [0.1, 0.2, 0.3]


def test_embed_passages_vide() -> None:
    em = EmbeddingModel(Settings())  # type: ignore[call-arg]
    assert em.embed_passages([]) == []


# --------------------------------------------------------------------------- #
# VectorStore                                                                 #
# --------------------------------------------------------------------------- #
def test_add_get_round_trip() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    store._collection = _FakeCollection()
    chunks = [
        _chunk(0, ClinicalSection.BIOLOGIE, title="Labs"),
        _chunk(1, ClinicalSection.MOTIF, title=None),
    ]
    store.add_chunks(chunks, [[0.0] * 3, [1.0] * 3])

    assert store.count() == 2
    restored = {c.chunk_id: c for c in store.get_all_chunks()}
    assert restored["doc::0000"].section is ClinicalSection.BIOLOGIE
    assert restored["doc::0000"].metadata["title"] == "Labs"
    # title None → stocké "" → restitué None.
    assert restored["doc::0001"].metadata["title"] is None


def test_upsert_idempotent() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    store._collection = _FakeCollection()
    c = _chunk(0, ClinicalSection.MOTIF)
    store.add_chunks([c], [[0.0] * 3])
    store.add_chunks([c], [[0.0] * 3])  # même id → pas de doublon
    assert store.count() == 1


def test_query_score_est_similarite() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    store._collection = _FakeCollection()
    store.add_chunks(
        [_chunk(0, ClinicalSection.CONCLUSION), _chunk(1, ClinicalSection.BIOLOGIE)],
        [[0.0] * 3, [1.0] * 3],
    )
    results = store.query([0.0, 0.0, 0.0], top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["doc::0000", "doc::0001"]
    # distances 0.1 et 0.2 → similarités 0.9 et 0.8.
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.8)
    assert all(r.retrieval_source == "vector" for r in results)


def test_add_chunks_longueur_incoherente() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    store._collection = _FakeCollection()
    with pytest.raises(ValueError):
        store.add_chunks([_chunk(0, ClinicalSection.MOTIF)], [[0.0] * 3, [1.0] * 3])


# --------------------------------------------------------------------------- #
# Gestion du corpus (label, list/delete/set_label) + round-trip label         #
# --------------------------------------------------------------------------- #
def _seed_two_docs(store: VectorStore) -> None:
    """Indexe 2 chunks pour « docA » (label cardio) et 1 pour « docB »."""
    store._collection = _FakeCollection()
    store.add_chunks(
        [
            _chunk(0, ClinicalSection.MOTIF, doc_id="docA", label="cardio"),
            _chunk(1, ClinicalSection.BIOLOGIE, doc_id="docA", label="cardio"),
            _chunk(0, ClinicalSection.MOTIF, doc_id="docB", label=""),
        ],
        [[0.0] * 3, [1.0] * 3, [0.5] * 3],
    )


def test_label_round_trip() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    store._collection = _FakeCollection()
    store.add_chunks(
        [_chunk(0, ClinicalSection.MOTIF, label="neuro")], [[0.0] * 3]
    )
    restored = store.get_all_chunks()[0]
    assert restored.metadata["label"] == "neuro"


def test_list_documents_agrege_par_doc() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    docs = {d["doc_id"]: d for d in store.list_documents()}

    assert docs["docA"]["n_chunks"] == 2
    assert docs["docA"]["label"] == "cardio"
    assert docs["docB"]["n_chunks"] == 1
    assert docs["docB"]["label"] == ""


def test_list_labels_distinct_non_vides() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    assert store.list_labels() == ["cardio"]  # "" écarté, dédoublonné


def test_query_where_filtre_label() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    results = store.query([0.0] * 3, top_k=10, where={"label": "cardio"})
    assert {r.chunk.doc_id for r in results} == {"docA"}
    assert len(results) == 2


def test_delete_document() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    n = store.delete_document("docA")
    assert n == 2
    assert store.count() == 1
    assert {c.doc_id for c in store.get_all_chunks()} == {"docB"}


def test_delete_document_inconnu() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    assert store.delete_document("absent") == 0
    assert store.count() == 3


def test_set_label() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    n = store.set_label("docB", "pneumo")
    assert n == 1
    docs = {d["doc_id"]: d for d in store.list_documents()}
    assert docs["docB"]["label"] == "pneumo"
    assert docs["docA"]["label"] == "cardio"  # inchangé


def test_set_label_inconnu() -> None:
    store = VectorStore(Settings())  # type: ignore[call-arg]
    _seed_two_docs(store)
    assert store.set_label("absent", "x") == 0
