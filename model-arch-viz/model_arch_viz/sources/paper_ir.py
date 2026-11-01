from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..formulas import io_for_kind, ops_for_kind
from ..ir import Architecture, Layer


def load_ir(path: str | Path) -> Architecture:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return architecture_from_dict(data)


def architecture_from_dict(data: dict[str, Any]) -> Architecture:
    if not isinstance(data, dict):
        raise TypeError("IR root must be a JSON object")
    if "name" not in data:
        raise ValueError("IR requires a top-level 'name'")
    arch = Architecture.from_dict(data)
    fill_missing_ops(arch)
    return arch


def fill_missing_ops(arch: Architecture) -> None:
    """Fill empty ops / io / notes from formula catalog using layer kind."""
    ctx = _ctx_from_dims(arch.dims)

    def visit(layer: Layer) -> None:
        catalog = {o.name: o for o in ops_for_kind(layer.kind, ctx)}
        if not layer.ops:
            layer.ops = list(catalog.values())
        else:
            filled = []
            for op in layer.ops:
                cat = catalog.get(op.name)
                note = op.note or (cat.note if cat else "")
                math = op.math or (cat.math if cat else op.math)
                filled.append(type(op)(name=op.name, math=math, note=note))
            layer.ops = filled
        if not layer.inputs and not layer.outputs:
            inputs, outputs = io_for_kind(layer.kind, ctx)
            layer.inputs = inputs
            layer.outputs = outputs
        for child in layer.children:
            visit(child)

    for block in arch.blocks:
        visit(block)


def _ctx_from_dims(dims: dict[str, str]) -> dict[str, Any]:
    """Prefer symbolic letters already used as keys."""
    ctx: dict[str, Any] = {}
    mapping = {
        "H": "hidden",
        "S": "seq",
        "B": "batch",
        "N": "heads",
        "D": "head_dim",
        "V": "vocab",
        "I": "intermediate",
        "E": "experts",
    }
    for letter, key in mapping.items():
        if letter in dims:
            ctx[key] = letter
    return ctx
