"""Recherche lexicale BM25 sur le corpus de chunks.

Responsabilité
--------------
Construire un index BM25 (in-memory) à partir des chunks et retourner les
`top_k` chunks les plus pertinents lexicalement pour une requête.

Pourquoi BM25 dans un pipeline RAG
----------------------------------
Sur du texte clinique, les termes exacts comptent : noms de molécules, codes
CIM, valeurs biologiques, abréviations. BM25 excelle sur ces correspondances
exactes là où les embeddings peuvent « lisser » le sens. La fusion RRF combine
ensuite ce signal lexical avec le signal sémantique.

Tokenisation
------------
Minuscules + extraction des tokens alphanumériques, en **préservant les nombres
décimaux** (« 5.2 » reste « 5.2 », utile pour les valeurs biologiques). L'index
est construit sur les mêmes `Chunk` que ceux stockés dans ChromaDB
(`VectorStore.get_all_chunks()`) → les deux recherches portent sur le même
corpus.
"""

from __future__ import annotations

import re

from models import Chunk, RetrievedChunk

# Tokens alphanumériques, avec point/virgule internes conservés (« 5.2 », « 1,5 »).
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][a-z0-9]+)*")


class BM25Index:
    """Index BM25 in-memory construit sur une liste de `Chunk`."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = None       # rank_bm25.BM25Okapi (construit paresseusement)
        self._tokenized: list[list[str]] | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _ensure_index(self):
        """Construit l'index BM25 au premier usage (import paresseux)."""
        if self._bm25 is None and self.chunks:
            from rank_bm25 import BM25Okapi

            self._tokenized = [self._tokenize(c.text) for c in self.chunks]
            self._bm25 = BM25Okapi(self._tokenized)
        return self._bm25

    def search(
        self, query: str, top_k: int, label: str | None = None
    ) -> list[RetrievedChunk]:
        """Retourne les `top_k` chunks les plus pertinents lexicalement.

        Seuls les chunks au score BM25 strictement positif sont renvoyés (les
        chunks sans recouvrement lexical avec la requête sont écartés). Si
        `label` est fourni, seuls les chunks portant ce label sont conservés
        (filtrage a posteriori sur l'index complet, déjà construit).
        """
        if not self.chunks:
            return []
        bm25 = self._ensure_index()
        scores = bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )

        results: list[RetrievedChunk] = []
        for i in ranked:
            score = float(scores[i])
            if score <= 0:
                break  # liste décroissante : plus rien de positif ensuite
            if label is not None and (
                self.chunks[i].metadata.get("label") or ""
            ) != label:
                continue  # hors du sous-corpus ciblé
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[i], score=score, retrieval_source="bm25"
                )
            )
            if len(results) >= top_k:
                break
        return results
