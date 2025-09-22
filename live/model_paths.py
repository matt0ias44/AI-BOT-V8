"""Utilities for locating model export directories used by the live stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def _as_path(candidate: str | Path, base_dir: Path) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_model_dir(
    base_dir: Path,
    env_var: str = "MODEL_DIR",
    candidates: Sequence[str | Path] | None = None,
) -> Path:
    """Return the first existing model directory from the provided candidates.

    Parameters
    ----------
    base_dir:
        Directory relative to which relative candidates will be resolved.
    env_var:
        Environment variable name that can override the search path.
    candidates:
        Ordered collection of directory names (relative or absolute) to try.
    """

    if candidates is None:
        candidates = ("models/bert_v7_1_plus", "models/bert_v7_1_multi")

    tried: list[Path] = []
    env_value = os.environ.get(env_var)
    if env_value:
        env_path = _as_path(env_value, base_dir)
        tried.append(env_path)
        if env_path.exists():
            return env_path
        print(
            f"[WARN] {env_var}={env_value!r} introuvable, tentative avec les répertoires par défaut..."
        )
    else:
        env_path = None

    for candidate in candidates:
        cand_path = _as_path(candidate, base_dir)
        tried.append(cand_path)
        if cand_path.exists():
            if env_path and cand_path != env_path:
                print(
                    f"[WARN] Utilisation du modèle de secours {cand_path} après échec sur {env_path}"
                )
            return cand_path

    tried_str = ", ".join(str(path) for path in tried)
    raise FileNotFoundError(
        f"Aucun répertoire modèle trouvé. Emplacements testés : {tried_str}"
    )
