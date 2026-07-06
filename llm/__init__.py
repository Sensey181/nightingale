"""Module LLM — génération augmentée, avec interface abstraite swappable.

Rôle dans le pipeline
---------------------
    (question + chunks retrievés)  ──▶  prompts.py       (prompt médical)
                                   ──▶  base.LLMClient    (interface abstraite)
                                   ──▶  réponse citée

Choix d'architecture
--------------------
- **Interface abstraite `LLMClient`** (`base.py`) : découple le pipeline du
  fournisseur de génération. Deux implémentations sont prévues :
    * `anthropic_client.AnthropicClient` — API Anthropic (claude-sonnet-4-6),
      pour la démo ;
    * un client Ollama local — pour la production HDS full local.
  Le swap ne touche qu'à un point d'injection (cf. scripts/migrate_to_local.py).

- **Prompt médical dédié** (`prompts.py`) : cadre la génération (répondre
  uniquement à partir du contexte fourni, citer les sources, signaler l'absence
  d'information) pour limiter les hallucinations sur un domaine sensible.

Note conformité HDS
-------------------
La partie génération est le seul maillon qui, en démo, envoie du contenu à une
API externe (Anthropic). C'est acceptable ici car le corpus est public et
anonymisé (MIMIC-IV). En production sur données patients réelles, on bascule
vers un LLM local (Ollama) : le retrieval étant déjà 100 % local, le pipeline
devient alors intégralement on-premise.
"""
