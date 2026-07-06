"""Prompt système et mise en forme du contexte pour la génération médicale.

Objectifs du prompt
-------------------
- **Ancrage strict** : répondre uniquement à partir des extraits fournis, ne
  pas inventer d'information clinique (réduction des hallucinations).
- **Citation des sources** : chaque affirmation renvoie au(x) chunk(s) source(s)
  pour la traçabilité — essentiel sur un domaine médical.
- **Aveu d'ignorance** : si l'information n'est pas dans le contexte, le dire
  explicitement plutôt que de combler.
- **Cadre non diagnostique** : le système assiste la recherche d'information
  dans des comptes-rendus, il ne pose pas de diagnostic.
"""

from __future__ import annotations

from models import RetrievedChunk

SYSTEM_PROMPT = (
    "Tu es un assistant d'aide à la lecture de comptes-rendus cliniques et de "
    "lettres de sortie. Tu réponds STRICTEMENT à partir des extraits fournis "
    "dans le contexte. Règles :\n"
    "1. N'utilise aucune connaissance externe ; appuie-toi uniquement sur le "
    "contexte.\n"
    "2. Cite systématiquement les sources sous la forme [source N].\n"
    "3. Si l'information demandée est absente du contexte, indique-le "
    "clairement.\n"
    "4. Tu n'établis pas de diagnostic : tu restitues et synthétises "
    "l'information présente dans les documents.\n"
    "5. Reste précis sur les valeurs, unités et posologies."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Met en forme les chunks retrievés en un bloc de contexte numéroté.

    Chaque source est préfixée par un identifiant [source N] réutilisable par le
    LLM pour la citation, et rappelle la section clinique (et le titre) d'origine.
    """
    if not chunks:
        return "(aucun extrait pertinent n'a été trouvé)"

    blocks: list[str] = []
    for i, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        title = chunk.metadata.get("title")
        label = chunk.section.value + (f" — {title}" if title else "")
        blocks.append(f"[source {i}] (section : {label})\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble le message utilisateur : contexte formaté + question + consigne."""
    return (
        "Contexte (extraits de comptes-rendus) :\n"
        f"{format_context(chunks)}\n\n"
        f"Question : {question}\n\n"
        "Réponds en français à partir du seul contexte ci-dessus, en citant les "
        "sources pertinentes sous la forme [source N]. Si l'information demandée "
        "est absente du contexte, indique-le clairement."
    )
