"""Interface Streamlit de démonstration de Nightingale.

Rôle
----
Interface web minimale pour la démo :
    1. l'utilisateur saisit une question médicale ;
    2. le `RetrievalPipeline` remonte les chunks pertinents (100 % local) ;
    3. l'`LLMClient` (Anthropic en démo) génère une réponse citée ;
    4. la réponse et les sources (chunks + section clinique) sont affichées.

Assemblage des composants
-------------------------
L'app câble `retrieval.pipeline.build_pipeline()` (BM25 + vectoriel + RRF +
reranking) et `llm.anthropic_client.AnthropicClient`. Les ressources lourdes
(index, modèles) sont mises en cache via `st.cache_resource` pour ne pas être
rechargées à chaque interaction.

Lancement
---------
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

# `streamlit run app/streamlit_app.py` place le dossier `app/` sur sys.path (et
# non la racine du projet). On ajoute la racine pour que les imports du projet
# (config, llm, retrieval) se résolvent quel que soit le répertoire de lancement.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corpus  # noqa: E402  (après l'ajout du sys.path)
from config import get_settings  # noqa: E402
from llm.base import LLMClient, LLMError
from llm.factory import build_llm
from retrieval.pipeline import RetrievalPipeline, build_pipeline

# Extensions acceptées à l'upload (miroir de ingestion._SUPPORTED_SUFFIXES).
_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]

# Modes de génération proposés dans l'UI : libellé lisible → (provider, aide).
_LLM_MODES = {
    "☁️ Cloud (Anthropic)": "anthropic",
    "🔒 Local (Ollama)": "ollama",
}


@st.cache_resource(show_spinner="Chargement de l'index et des modèles…")
def _load_pipeline() -> RetrievalPipeline:
    """Assemble le pipeline de retrieval (mis en cache entre les reruns)."""
    return build_pipeline()


@st.cache_resource(show_spinner=False)
def _load_llm(provider: str) -> LLMClient:
    """Instancie le client de génération pour le provider (mis en cache par provider)."""
    return build_llm(provider)


# Libellés lisibles de l'origine d'un chunk (champ `retrieval_source`), pour
# expliciter dans l'UI pourquoi un extrait a été remonté.
_SOURCE_LABELS = {
    "bm25": "BM25 (lexical)",
    "vector": "vectoriel (sémantique)",
    "rrf": "fusion RRF",
    "reranked": "reranké",
    "hybrid": "hybride",
}

# Repère une citation « [source N] » (tolère « [sources 1 et 2] », « [source 1, 3] »)
# produite par le LLM, pour la transformer en lien cliquable vers l'extrait.
_CITATION_RE = re.compile(r"\[\s*sources?\s+([^\]]+?)\s*\]", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")


def _linkify_citations(answer: str, n_sources: int) -> str:
    """Transforme les « [source N] » de la réponse en liens ancrés « #source-N ».

    Seuls les numéros valides (1..n_sources) sont convertis ; le reste est laissé
    tel quel. Une citation multiple « [source 1, 3] » devient deux liens.
    """

    def _replace(match: re.Match) -> str:
        numbers = [int(n) for n in _NUM_RE.findall(match.group(1))]
        valid = [n for n in numbers if 1 <= n <= n_sources]
        if not valid:
            return match.group(0)  # citation hors périmètre : inchangée
        return " ".join(f"[[source {n}]](#source-{n})" for n in valid)

    return _CITATION_RE.sub(_replace, answer)


def _render_sources(chunks) -> None:
    """Affiche les extraits sources sous la réponse, dépliés et traçables.

    Chaque source porte une ancre HTML « source-N » ciblée par les citations
    « [source N] » de la réponse (voir `_linkify_citations`).
    """
    st.subheader(f"Sources ({len(chunks)})", anchor=False)
    st.caption(
        "Extraits ayant servi à la réponse, du plus au moins pertinent. "
        "Les citations [source N] ci-dessus renvoient ici."
    )
    for i, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        title = chunk.metadata.get("title")
        section_label = chunk.section.value + (f" — {title}" if title else "")
        origin = _SOURCE_LABELS.get(
            retrieved.retrieval_source, retrieved.retrieval_source
        )
        header = (
            f"source {i} · {chunk.doc_id} · {section_label} "
            f"· score {retrieved.score:.3f}"
        )
        # Ancre invisible ciblée par les liens de citation de la réponse.
        st.markdown(f"<span id='source-{i}'></span>", unsafe_allow_html=True)
        with st.expander(header, expanded=True):
            st.caption(
                f"document : `{chunk.doc_id}` · section : {section_label} "
                f"· position : {chunk.position} · origine : {origin}"
            )
            st.write(chunk.text)


def _safe_count(store) -> int:
    """Nombre de chunks indexés, robuste à un index absent/illisible."""
    try:
        return store.count()
    except Exception as exc:  # index absent / illisible
        st.error(f"Index illisible : {exc}")
        return 0


def _provider_selector(settings) -> str:
    """Bascule Cloud/Local du modèle de génération. Retourne le provider choisi."""
    labels = list(_LLM_MODES)
    default = "🔒 Local (Ollama)" if settings.llm_provider == "ollama" else labels[0]
    chosen = st.segmented_control(
        "Modèle de génération",
        labels,
        default=default,
        selection_mode="single",
    )
    provider = _LLM_MODES.get(chosen or default, "anthropic")
    model = settings.ollama_model if provider == "ollama" else settings.llm_model
    note = (
        "génération en local."
        if provider == "ollama"
        else "génération via l'API Anthropic (démo)"
    )
    st.caption(f"Modèle : `{model}` — {note}.")
    return provider


def _render_search_tab(pipeline: RetrievalPipeline, store, settings) -> None:
    """Onglet recherche : mode LLM + filtre label + question → réponse citée."""
    provider = _provider_selector(settings)

    label_filter: str | None = None
    labels = store.list_labels()
    if labels:
        label_filter = st.selectbox(
            "Filtrer par label",
            labels,
            index=None,
            placeholder="Tous les labels",
        )

    question = st.text_input(
        "Question médicale",
        placeholder="Ex. : Quel est le motif d'hospitalisation ?",
    )
    if not question:
        return

    with st.spinner("Recherche des extraits pertinents…"):
        chunks = pipeline.retrieve(question, label=label_filter)

    if not chunks:
        msg = "Aucun extrait pertinent trouvé pour cette question."
        if label_filter:
            msg += " Essayez sans filtre de label."
        st.info(msg)
        return

    with st.spinner("Génération de la réponse…"):
        try:
            answer = _load_llm(provider).generate(question, chunks)
        except LLMError as exc:
            st.error(f"Génération impossible : {exc}")
            return

    st.subheader("Réponse", anchor=False)
    st.markdown(_linkify_citations(answer, len(chunks)))
    st.divider()
    _render_sources(chunks)


def _chunk_count_selector(total: int, *, key: str) -> int:
    """Curseur + champ numérique synchronisés pour le nombre de chunks à afficher.

    Les deux widgets partagent la même valeur (callbacks croisés), bornée à
    `total`, par pas de 10, avec un défaut de 20. Retourne le nombre choisi.
    """
    sl_key, num_key = f"{key}_sl", f"{key}_num"
    if sl_key not in st.session_state:
        st.session_state[sl_key] = min(20, total)
        st.session_state[num_key] = min(20, total)

    def _from_slider() -> None:
        st.session_state[num_key] = st.session_state[sl_key]

    def _from_num() -> None:
        st.session_state[sl_key] = st.session_state[num_key]

    col_slider, col_num = st.columns([3, 1], vertical_alignment="center")
    with col_slider:
        st.slider(
            "Nombre de chunks à afficher",
            min_value=10,
            max_value=total,
            step=10,
            key=sl_key,
            on_change=_from_slider,
        )
    with col_num:
        st.number_input(
            "Nombre",
            min_value=10,
            max_value=total,
            step=10,
            key=num_key,
            on_change=_from_num,
        )
    return int(st.session_state[sl_key])


def _render_document_chunks(store, doc_id: str, *, limit: int | None = None) -> None:
    """Liste les chunks d'un document, ordonnés par position (bornés par `limit`)."""
    doc_chunks = sorted(
        (c for c in store.get_all_chunks() if c.doc_id == doc_id),
        key=lambda c: c.position,
    )
    if limit is not None:
        doc_chunks = doc_chunks[:limit]
    for chunk in doc_chunks:
        title = chunk.metadata.get("title")
        section_label = chunk.section.value + (f" — {title}" if title else "")
        st.markdown(f"**position {chunk.position}** · {section_label}")
        st.write(chunk.text)


def _commit_new_label(key: str) -> None:
    """Callback de validation (Enter) de la barre « Nouveau label ».

    Intègre le label saisi à la liste des options, le sélectionne dans le
    selectbox, referme la barre de saisie et la vide — pour éviter la redondance
    visuelle (le label n'existe plus qu'à un seul endroit : la barre principale).
    """
    value = st.session_state.get(f"{key}_new", "").strip()
    if value:
        st.session_state[f"{key}_pending"] = value
        st.session_state[f"{key}_sel"] = value  # présélectionne le nouveau label
    st.session_state[f"{key}_addmode"] = False
    st.session_state[f"{key}_new"] = ""


def _label_picker(store, *, current: str, key: str) -> str:
    """Widget de choix de label : sélection d'un label existant, ou création.

    La barre de sélection ne propose que les labels déjà présents
    (`store.list_labels()`, pour éviter les doublons de casse « cardio » vs
    « Cardio »), plus tout label fraîchement saisi. Un bouton « Nouveau label »
    aligné à côté révèle une barre de saisie ; valider par Entrée intègre le
    label à la liste et referme la barre (voir `_commit_new_label`). Retourne le
    label choisi ("" si aucun).
    """
    labels = list(store.list_labels())
    pending = st.session_state.get(f"{key}_pending", "")
    options = labels + ([pending] if pending and pending not in labels else [])

    col_sel, col_add = st.columns([2, 1], vertical_alignment="bottom")
    with col_sel:
        index = options.index(current) if current in options else None
        choice = st.selectbox(
            "Label",
            options,
            index=index,
            placeholder="Aucun label",
            key=f"{key}_sel",
        )
    with col_add:
        if st.button(
            "Nouveau label", key=f"{key}_addbtn", use_container_width=True
        ):
            st.session_state[f"{key}_addmode"] = True

    if st.session_state.get(f"{key}_addmode"):
        st.text_input(
            "Nouveau label",
            key=f"{key}_new",
            on_change=_commit_new_label,
            args=(key,),
        )

    return choice or ""


@st.dialog("Réinitialiser l'index")
def _reset_index_dialog(store) -> None:
    st.write(
        "Supprime **tous** les chunks de la base vectorielle. Les documents "
        "traités (`data/processed`) sont conservés : une réindexation les "
        "restaure."
    )
    if st.button("Réinitialiser", type="primary"):
        store.reset()
        st.cache_resource.clear()
        st.session_state["_toast"] = "Index réinitialisé"
        st.rerun()


@st.dialog("Supprimer le document")
def _delete_doc_dialog(doc_id: str, settings, store) -> None:
    st.write(
        f"Le document « {doc_id} » et ses chunks seront retirés de l'index. "
        "Le fichier brut d'origine (`data/raw`) est conservé."
    )
    if st.button("Supprimer", type="primary"):
        corpus.delete_document(doc_id, settings=settings, store=store)
        st.cache_resource.clear()
        st.rerun()


@st.dialog("Modifier le label")
def _edit_label_dialog(doc_id: str, current: str, settings, store) -> None:
    new_label = _label_picker(store, current=current, key=f"edit_{doc_id}")
    if st.button("Enregistrer", type="primary"):
        corpus.set_document_label(
            doc_id, new_label, settings=settings, store=store
        )
        st.cache_resource.clear()
        st.rerun()


@st.dialog("Retirer le label")
def _remove_label_dialog(doc_id: str, label: str, settings, store) -> None:
    st.write(
        f"Le label « {label} » sera retiré du document « {doc_id} ». Il ne "
        "sera plus filtrable par ce label."
    )
    if st.button("Retirer", type="primary"):
        corpus.set_document_label(doc_id, "", settings=settings, store=store)
        st.cache_resource.clear()
        st.rerun()


def _render_index_actions(store) -> None:
    """Actions globales sur l'index : recharger, réinitialiser (via dialog)."""
    st.subheader("Index", anchor=False)
    col_reload, col_reset = st.columns(2)
    with col_reload:
        if st.button(
            "Recharger l'index",
            use_container_width=True,
            help=(
                "Relit l'index depuis le disque en vidant le cache. À utiliser "
                "après une réindexation hors de l'app (`python -m ingestion` "
                "puis `python -m indexing --reset`)."
            ),
        ):
            st.cache_resource.clear()
            st.session_state["_toast"] = "Index rechargé"
            st.rerun()
    with col_reset:
        if st.button("Réinitialiser l'index", use_container_width=True):
            _reset_index_dialog(store)


def _render_corpus_tab(store, settings) -> None:
    """Onglet corpus : total, liste, actions d'index, labellisation, suppression."""
    documents = corpus.list_documents(store)
    total_chunks = sum(d["n_chunks"] for d in documents)
    st.metric("Chunks indexés au total", total_chunks)

    st.subheader("Documents indexés", anchor=False)
    if not documents:
        st.info("Aucun document indexé pour l'instant.")
        st.divider()
        _render_index_actions(store)
        return

    st.dataframe(
        [
            {
                "doc_id": d["doc_id"],
                "titre": d["title"] or "—",
                "label": d["label"] or "—",
                "chunks": d["n_chunks"],
            }
            for d in documents
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    _render_index_actions(store)

    st.divider()
    st.subheader("Gérer un document", anchor=False)
    doc_ids = [d["doc_id"] for d in documents]
    selected = st.selectbox("Document sélectionné", doc_ids)
    current = next(d for d in documents if d["doc_id"] == selected)

    st.markdown(f"**Label :** {current['label'] or '—'}")
    col_edit, col_remove = st.columns(2)
    with col_edit:
        if st.button(
            "Modifier le label", use_container_width=True, key=f"edit_{selected}"
        ):
            _edit_label_dialog(selected, current["label"], settings, store)
    with col_remove:
        if st.button(
            "Retirer le label",
            use_container_width=True,
            disabled=not current["label"],
            key=f"rmlabel_{selected}",
        ):
            _remove_label_dialog(selected, current["label"], settings, store)

    if st.button(
        "Supprimer le document",
        type="primary",
        use_container_width=True,
        key=f"del_{selected}",
    ):
        _delete_doc_dialog(selected, settings, store)

    st.subheader(f"Chunks de « {selected} »", anchor=False)
    total = current["n_chunks"]
    if total > 10:
        n_show = _chunk_count_selector(total, key=f"chunks_{selected}")
        st.caption(f"{min(n_show, total)} chunks affichés sur {total}.")
    else:
        n_show = total
        st.caption(
            f"Ce document compte {total} chunk{'s' if total > 1 else ''} — "
            "tous affichés."
        )
    _render_document_chunks(store, selected, limit=n_show)


def _render_add_tab(store, settings) -> None:
    """Onglet d'ajout : upload → OCR → indexation, avec progression détaillée."""
    st.subheader("Ajouter un document", anchor=False)
    st.caption(
        "Fichier clinique (PDF ou image). OCR local (MinerU) puis indexation."
    )
    uploaded = st.file_uploader(
        "Fichier clinique (PDF / image)", type=_UPLOAD_TYPES
    )
    add_label = _label_picker(store, current="", key="add_label")

    if uploaded is None or not st.button("Ingérer et indexer", type="primary"):
        return

    raw_dir = Path(settings.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / uploaded.name
    dest.write_bytes(uploaded.getbuffer())

    log: list[str] = []
    with st.status("Traitement du document…", expanded=True) as status:
        console = st.empty()

        def on_step(message: str) -> None:
            log.append(f"$ {message}")
            console.code("\n".join(log), language="text")

        on_step(f"Fichier reçu : {uploaded.name}")
        try:
            doc_id, n_chunks = corpus.add_document(
                dest,
                label=add_label,
                settings=settings,
                store=store,
                on_step=on_step,
            )
        except Exception as exc:  # OCR/index : remonter l'erreur à l'UI
            on_step(f"✗ Échec : {exc}")
            status.update(label="Échec de l'ingestion", state="error")
            return

        status.update(
            label=f"Document « {doc_id} » indexé ({n_chunks} chunks)",
            state="complete",
        )

    st.success(
        f"Document « {doc_id} » indexé ({n_chunks} chunks). "
        "Il apparaît désormais dans l'onglet **Corpus**."
    )
    st.cache_resource.clear()


def main() -> None:
    settings = get_settings()

    st.set_page_config(page_title="Nightingale", page_icon="🩺", layout="wide")
    # Masque les boutons +/- (steppers) des champs numériques.
    st.markdown(
        "<style>"
        "[data-testid='stNumberInputStepUp'],"
        "[data-testid='stNumberInputStepDown']{display:none;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("Nightingale", anchor=False)
    st.caption(
        "RAG local sur comptes-rendus cliniques"
    )

    # Retour visuel des actions ayant déclenché un rerun (recharger/réinitialiser).
    if "_toast" in st.session_state:
        st.toast(st.session_state.pop("_toast"), icon="✅")

    pipeline = _load_pipeline()
    store = pipeline.vector_search.store
    n_chunks = _safe_count(store)

    tab_search, tab_corpus, tab_add = st.tabs(
        ["🔎 Recherche", "🗂️ Corpus", "➕ Ajouter un document"]
    )
    with tab_search:
        if not n_chunks:
            st.warning(
                "L'index est vide. Ajoutez un document depuis l'onglet "
                "**Ajouter un document**."
            )
        else:
            _render_search_tab(pipeline, store, settings)
    with tab_corpus:
        _render_corpus_tab(store, settings)
    with tab_add:
        _render_add_tab(store, settings)


if __name__ == "__main__":
    main()
