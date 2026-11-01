from __future__ import annotations

from pathlib import Path

from .ir import Architecture
from .sources.hf_config import from_config_file, from_hf as _from_hf_config
from .sources.hf_meta import enrich_with_meta
from .sources.paper_ir import load_ir


def from_hf(
    model_id: str,
    *,
    meta: bool = False,
    revision: str | None = None,
    token: str | None = None,
) -> Architecture:
    arch = _from_hf_config(model_id, revision=revision, token=token)
    if meta:
        arch = enrich_with_meta(
            arch, model_id, revision=revision, token=token
        )
    return arch


def from_ir(path: str | Path) -> Architecture:
    return load_ir(path)


def from_config_json(path: str | Path) -> Architecture:
    return from_config_file(path)
