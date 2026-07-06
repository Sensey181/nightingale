"""Tests du retrieval : BM25, RRF, reranker, recherche vectorielle, pipeline.

La fusion RRF et la tokenisation BM25 sont testées directement. Les composants
à dépendances lourdes (rank_bm25, cross-encoder, embeddings, ChromaDB) sont
remplacés par des doublures injectées.
"""

from __future__ import annotations

from config import Settings
from models import Chunk, ClinicalSection, RetrievedChunk
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import Reranker
from retrieval.vector_search import VectorSearch


def _chunk(cid: str, text: str = "texte", label: str = "") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc",
        text=text,
        section=ClinicalSection.AUTRE,
        position=0,
        metadata={"label": label},
    )


def _rc(cid: str, score: float = 1.0, text: str = "texte") -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(cid, text), score=score, retrieval_source="x")


# --------------------------------------------------------------------------- #
# fusion RRF                                                                  #
# --------------------------------------------------------------------------- #
def test_rrf_combine_et_dedoublonne() -> None:
    bm25 = [_rc("a"), _rc("b"), _rc("c")]
    vect = [_rc("b"), _rc("a"), _rc("d")]
    fused = reciprocal_rank_fusion([bm25, vect], k=60)

    ids = [r.chunk.chunk_id for r in fused]
    # a et b apparaissent dans les deux listes → devant.
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids) == {"a", "b", "c", "d"}  # dédoublonné
    assert all(r.retrieval_source == "rrf" for r in fused)
    # Scores strictement décroissants.
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_score_exact() -> None:
    fused = reciprocal_rank_fusion([[_rc("a"), _rc("b")], [_rc("a")]], k=60)
    by_id = {r.chunk.chunk_id: r.score for r in fused}
    # a : rang 1 dans les deux listes → 1/61 + 1/61.
    assert by_id["a"] == 2 / 61
    # b : rang 2 dans une seule liste → 1/62.
    assert by_id["b"] == 1 / 62


def test_rrf_listes_vides() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


# --------------------------------------------------------------------------- #
# BM25                                                                        #
# --------------------------------------------------------------------------- #
def test_bm25_tokenise_conserve_decimales() -> None:
    assert BM25Index._tokenize("Na 5.2 mmol/L") == ["na", "5.2", "mmol", "l"]


def test_bm25_search_wiring() -> None:
    chunks = [_chunk("a", "aspirine"), _chunk("b", "metoprolol"), _chunk("c", "na")]
    index = BM25Index(chunks)

    class _FakeBM25:
        def get_scores(self, query_tokens):
            # b le plus pertinent, a ensuite, c non pertinent (0).
            return [0.5, 0.9, 0.0]

    index._bm25 = _FakeBM25()  # court-circuite rank_bm25
    results = index.search("query", top_k=5)

    # Score 0 filtré ; tri décroissant.
    assert [r.chunk.chunk_id for r in results] == ["b", "a"]
    assert results[0].score == 0.9
    assert all(r.retrieval_source == "bm25" for r in results)


def test_bm25_corpus_vide() -> None:
    assert BM25Index([]).search("q", top_k=5) == []


def test_bm25_filtre_label() -> None:
    chunks = [
        _chunk("a", "aspirine", label="cardio"),
        _chunk("b", "aspirine", label="neuro"),
        _chunk("c", "aspirine", label="cardio"),
    ]
    index = BM25Index(chunks)

    class _FakeBM25:
        def get_scores(self, query_tokens):
            return [0.9, 0.8, 0.7]  # a, b, c tous pertinents

    index._bm25 = _FakeBM25()
    results = index.search("aspirine", top_k=5, label="cardio")

    # Seuls les chunks du label ciblé sont conservés (b « neuro » écarté).
    assert [r.chunk.chunk_id for r in results] == ["a", "c"]


# --------------------------------------------------------------------------- #
# reranker                                                                    #
# --------------------------------------------------------------------------- #
def test_reranker_reordonne_et_tronque() -> None:
    reranker = Reranker(Settings())  # type: ignore[call-arg]

    class _FakeCE:
        def predict(self, pairs):
            # Scores dans l'ordre des candidats a,b,c → c meilleur, puis a, puis b.
            return [0.2, 0.1, 0.9]

    reranker._model = _FakeCE()
    candidates = [_rc("a"), _rc("b"), _rc("c")]
    out = reranker.rerank("query", candidates, top_n=2)

    assert [r.chunk.chunk_id for r in out] == ["c", "a"]
    assert out[0].score == 0.9
    assert all(r.retrieval_source == "reranked" for r in out)


def test_reranker_candidats_vides() -> None:
    assert Reranker(Settings()).rerank("q", [], top_n=5) == []  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# vector_search                                                               #
# --------------------------------------------------------------------------- #
def test_vector_search_delegue() -> None:
    class _FakeEmbedder:
        def embed_query(self, text):
            assert text == "douleur"
            return [0.1, 0.2]

    class _FakeStore:
        def query(self, emb, top_k):
            assert emb == [0.1, 0.2] and top_k == 3
            return [_rc("a")]

    vs = VectorSearch(_FakeEmbedder(), _FakeStore())
    results = vs.search("douleur", top_k=3)
    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_vector_search_filtre_label() -> None:
    class _FakeEmbedder:
        def embed_query(self, text):
            return [0.1, 0.2]

    class _FakeStore:
        def query(self, emb, top_k, where=None):
            # Le label est transmis comme filtre de métadonnées ChromaDB.
            assert where == {"label": "cardio"}
            return [_rc("a")]

    vs = VectorSearch(_FakeEmbedder(), _FakeStore())
    results = vs.search("douleur", top_k=3, label="cardio")
    assert [r.chunk.chunk_id for r in results] == ["a"]


# --------------------------------------------------------------------------- #
# pipeline (orchestration)                                                    #
# --------------------------------------------------------------------------- #
def test_pipeline_orchestration() -> None:
    calls = {}

    class _FakeBM25:
        def search(self, query, top_k, label=None):
            calls["bm25"] = (query, top_k, label)
            return [_rc("a"), _rc("b")]

    class _FakeVS:
        def search(self, query, top_k, label=None):
            calls["vector"] = (query, top_k, label)
            return [_rc("b"), _rc("c")]

    class _FakeReranker:
        def rerank(self, query, candidates, top_n):
            calls["rerank"] = (query, [c.chunk.chunk_id for c in candidates], top_n)
            return candidates[:top_n]

    settings = Settings(  # type: ignore[call-arg]
        top_k_bm25=7, top_k_vector=9, rrf_k=60, top_n_rerank=2
    )
    pipeline = RetrievalPipeline(_FakeBM25(), _FakeVS(), _FakeReranker(), settings)
    results = pipeline.retrieve("ma question", label="cardio")

    # Chaque étape reçoit les bons paramètres de config + le filtre label.
    assert calls["bm25"] == ("ma question", 7, "cardio")
    assert calls["vector"] == ("ma question", 9, "cardio")
    # Le reranker reçoit les candidats fusionnés (dédoublonnés) et top_n.
    assert calls["rerank"][0] == "ma question"
    assert calls["rerank"][2] == 2
    assert set(calls["rerank"][1]) == {"a", "b", "c"}
    assert len(results) == 2
