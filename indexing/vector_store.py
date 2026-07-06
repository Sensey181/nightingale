"""Stockage et interrogation vectorielle via ChromaDB (local, persisté).

Responsabilité
--------------
- Créer/ouvrir une collection ChromaDB persistée sur disque (`data/chroma`).
- Indexer des `Chunk` avec leurs embeddings et métadonnées (doc_id, section…).
- Exposer une recherche par similarité (top-k) pour le retrieval vectoriel.
- Exposer l'ensemble des chunks (`get_all_chunks`) pour (re)construire l'index
  BM25 côté retrieval — les deux recherches portent ainsi sur le même corpus.

Choix d'architecture
--------------------
ChromaDB en mode `PersistentClient` : l'index vit entièrement sur la machine,
aucun appel réseau → conforme aux exigences HDS côté données. Espace de
similarité **cosinus** (cohérent avec les embeddings e5 normalisés). L'import de
`chromadb` est paresseux pour garder le module importable sans la dépendance.

Contrainte ChromaDB : les valeurs de métadonnées doivent être des scalaires
(str/int/float/bool). Le titre `None` est stocké comme chaîne vide et restitué
en `None`.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings
from models import Chunk, ClinicalSection, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper autour d'une collection ChromaDB persistée localement."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------ #
    # Cycle de vie                                                       #
    # ------------------------------------------------------------------ #
    def _ensure_client(self):
        if self._client is None:
            import chromadb

            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.settings.chroma_dir)
            )
        return self._client

    def _ensure_collection(self):
        if self._collection is None:
            client = self._ensure_client()
            self._collection = client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset(self) -> None:
        """Supprime la collection (réindexation propre)."""
        client = self._ensure_client()
        try:
            client.delete_collection(self.settings.chroma_collection)
        except Exception:  # collection inexistante : rien à supprimer
            logger.debug("Collection déjà absente lors du reset.")
        self._collection = None

    def count(self) -> int:
        """Nombre de chunks indexés."""
        return self._ensure_collection().count()

    # ------------------------------------------------------------------ #
    # Indexation                                                         #
    # ------------------------------------------------------------------ #
    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Indexe (upsert) des chunks + embeddings dans la collection.

        L'upsert rend l'opération idempotente : réindexer un chunk déjà présent
        met à jour son contenu plutôt que de le dupliquer.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks et embeddings doivent avoir la même longueur."
            )
        collection = self._ensure_collection()
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[self._to_metadata(c) for c in chunks],
        )

    # ------------------------------------------------------------------ #
    # Recherche                                                          #
    # ------------------------------------------------------------------ #
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Recherche les `top_k` chunks les plus proches d'un embedding requête.

        `where` : filtre de métadonnées ChromaDB optionnel (ex.
        `{"label": "cardio"}`) pour restreindre la recherche à un sous-corpus.
        """
        collection = self._ensure_collection()
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = collection.query(**kwargs)
        return self._parse_query_result(result)

    def get_all_chunks(self) -> list[Chunk]:
        """Retourne tous les chunks indexés (pour reconstruire l'index BM25)."""
        collection = self._ensure_collection()
        result = collection.get(include=["documents", "metadatas"])
        return [
            self._to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    # ------------------------------------------------------------------ #
    # Gestion du corpus (niveau document)                                #
    # ------------------------------------------------------------------ #
    def list_documents(self) -> list[dict]:
        """Agrège les chunks indexés par document.

        Returns:
            Une liste de dicts `{doc_id, n_chunks, title, label}`, triée par
            `doc_id` — de quoi peupler un tableau de gestion du corpus.
        """
        collection = self._ensure_collection()
        result = collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in result["metadatas"]:
            doc_id = meta.get("doc_id", "")
            entry = docs.setdefault(
                doc_id,
                {"doc_id": doc_id, "n_chunks": 0, "title": None, "label": ""},
            )
            entry["n_chunks"] += 1
            if not entry["title"] and meta.get("title"):
                entry["title"] = meta.get("title")
            if not entry["label"] and meta.get("label"):
                entry["label"] = meta.get("label")
        return sorted(docs.values(), key=lambda d: d["doc_id"])

    def list_labels(self) -> list[str]:
        """Retourne l'ensemble trié des labels non vides présents dans l'index."""
        collection = self._ensure_collection()
        result = collection.get(include=["metadatas"])
        labels = {(m.get("label") or "") for m in result["metadatas"]}
        labels.discard("")
        return sorted(labels)

    def delete_document(self, doc_id: str) -> int:
        """Supprime tous les chunks d'un document. Retourne le nombre supprimé."""
        collection = self._ensure_collection()
        ids = collection.get(where={"doc_id": doc_id}).get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def set_label(self, doc_id: str, label: str) -> int:
        """Met à jour le label de tous les chunks d'un document (in place).

        Retourne le nombre de chunks mis à jour (0 si document inconnu).
        """
        collection = self._ensure_collection()
        got = collection.get(where={"doc_id": doc_id})
        ids = got.get("ids", [])
        if not ids:
            return 0
        metadatas = got.get("metadatas", [])
        for meta in metadatas:
            meta["label"] = label or ""
        collection.update(ids=ids, metadatas=metadatas)
        return len(ids)

    # ------------------------------------------------------------------ #
    # (dé)sérialisation                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_metadata(chunk: Chunk) -> dict:
        return {
            "doc_id": chunk.doc_id,
            "section": chunk.section.value,
            "position": chunk.position,
            "title": chunk.metadata.get("title") or "",
            "label": chunk.metadata.get("label") or "",
        }

    @staticmethod
    def _to_chunk(chunk_id: str, document: str, metadata: dict) -> Chunk:
        try:
            section = ClinicalSection(metadata.get("section"))
        except ValueError:
            section = ClinicalSection.AUTRE
        return Chunk(
            chunk_id=chunk_id,
            doc_id=metadata.get("doc_id", ""),
            text=document,
            section=section,
            position=int(metadata.get("position", 0)),
            metadata={
                "title": metadata.get("title") or None,
                "label": metadata.get("label") or "",
            },
        )

    def _parse_query_result(self, result: dict) -> list[RetrievedChunk]:
        # ChromaDB renvoie des listes de listes (une par requête) ; on n'envoie
        # qu'une requête → index 0.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            chunk = self._to_chunk(chunk_id, document, metadata)
            # Distance cosinus ∈ [0, 2] → similarité = 1 - distance.
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0 - float(distance),
                    retrieval_source="vector",
                )
            )
        return retrieved
