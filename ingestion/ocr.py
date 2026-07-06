"""OCR des documents cliniques via MinerU en local (in-process).

Responsabilité
--------------
Transcrire un fichier (PDF scanné, image) en Markdown structuré (titres, listes,
tableaux préservés) en appelant le moteur MinerU **installé localement**. Le
Markdown est ensuite exploité par `parser.py` pour retrouver la structure
clinique.

Pourquoi MinerU en local
------------------------
- **100 % local, sans GPU** : le backend `pipeline` de MinerU tourne en pur CPU
  (accuracy ~86), ce qui répond à la contrainte « machine sans GPU » et à
  l'argument de conformité HDS — aucun document ne sort de la machine.
- Bonne restitution de la structure documentaire (utile pour le chunking
  sémantique par section).

Fonctionnement
--------------
On s'appuie sur l'API in-process de MinerU (`mineru.cli.common`) :

1. `read_fn(path)`   → lit les octets du fichier (convertit une image en PDF).
2. `do_parse(...)`   → exécute l'OCR/extraction et écrit les résultats sur disque
   dans un répertoire temporaire, dont `{nom}.md`.
3. On relit le fichier Markdown produit et on le renvoie.

Le premier appel télécharge les modèles MinerU (mise en cache locale ensuite).
En cas de souci de téléchargement, `mineru_model_source="modelscope"` bascule
sur un miroir sans proxy.

Note d'installation : `mineru` (et ses dépendances lourdes : torch, etc.) n'est
pas installé dans le python système. L'import est donc **paresseux** — le module
reste importable pour les tests, qui injectent des doublures.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class MinerUError(RuntimeError):
    """Erreur remontée par le client MinerU (fichier, moteur, ou sortie OCR)."""


def _load_mineru():
    """Importe paresseusement l'API in-process de MinerU.

    Isolé dans une fonction pour (1) garder `ocr.py` importable sans MinerU
    installé et (2) offrir un point de monkeypatch simple aux tests.

    Returns:
        Le couple `(do_parse, read_fn)` de `mineru.cli.common`.
    """
    try:
        from mineru.cli.common import do_parse, read_fn
    except ImportError as exc:  # dépendance non installée
        raise MinerUError(
            "MinerU n'est pas installé : `uv pip install -U \"mineru[core]\"`."
        ) from exc
    return do_parse, read_fn


class MinerUClient:
    """Wrapper autour du moteur MinerU local (backend CPU par défaut).

    Le nom « Client » est conservé pour compatibilité avec le pipeline, mais il
    n'y a plus aucun appel réseau : tout se passe en local, in-process.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Miroir de téléchargement des modèles, si configuré (ex. "modelscope").
        if self.settings.mineru_model_source:
            os.environ.setdefault(
                "MINERU_MODEL_SOURCE", self.settings.mineru_model_source
            )

    # ------------------------------------------------------------------ #
    # API publique                                                       #
    # ------------------------------------------------------------------ #
    def ocr_to_markdown(self, file_path: Path) -> str:
        """Transcrit un fichier en Markdown structuré via MinerU local.

        Args:
            file_path: chemin du PDF/image à transcrire.

        Returns:
            Le contenu Markdown produit par MinerU.

        Raises:
            MinerUError: fichier introuvable, MinerU absent, échec de
                l'extraction, ou aucun Markdown en sortie.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise MinerUError(f"Fichier introuvable : {file_path}")

        do_parse, read_fn = _load_mineru()

        try:
            pdf_bytes = read_fn(file_path)
        except Exception as exc:  # lecture / conversion image → pdf
            raise MinerUError(f"Lecture de {file_path.name} échouée : {exc}") from exc

        name = file_path.stem
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
            logger.info(
                "MinerU (local, backend=%s) : extraction de %s",
                self.settings.mineru_backend,
                file_path.name,
            )
            try:
                do_parse(
                    output_dir=tmp,
                    pdf_file_names=[name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=[self.settings.mineru_language],
                    backend=self.settings.mineru_backend,
                    parse_method=self.settings.mineru_parse_method,
                    formula_enable=self.settings.mineru_enable_formula,
                    table_enable=self.settings.mineru_enable_table,
                    # On ne veut que le Markdown : on coupe les sorties annexes
                    # (JSON intermédiaires, PDF d'origine, calques de debug).
                    f_dump_md=True,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_orig_pdf=False,
                    f_dump_content_list=False,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                )
            except Exception as exc:  # échec du moteur d'extraction
                raise MinerUError(
                    f"Extraction MinerU échouée pour {file_path.name} : {exc}"
                ) from exc

            return self._read_markdown(Path(tmp), name)

    # ------------------------------------------------------------------ #
    # Étapes internes                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_markdown(output_dir: Path, name: str) -> str:
        """Relit le Markdown écrit par MinerU sous `output_dir`.

        MinerU écrit dans `output_dir/{name}/{méthode}/{name}.md`. On localise
        le `.md` par glob (la méthode réelle — `ocr`/`txt` — dépend du contenu),
        en préférant le fichier `{name}.md`.
        """
        md_files = sorted(output_dir.rglob("*.md"))
        if not md_files:
            raise MinerUError(
                f"MinerU n'a produit aucun Markdown pour {name} "
                f"(répertoire de sortie vide)."
            )
        chosen = next(
            (p for p in md_files if p.name == f"{name}.md"), md_files[0]
        )
        return chosen.read_text(encoding="utf-8")
