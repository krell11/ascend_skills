from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class TensorSpec:
    name: str
    shape: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(name=str(data["name"]), shape=[str(s) for s in data.get("shape", [])])


@dataclass
class Op:
    name: str
    math: str
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Op:
        return cls(
            name=str(data["name"]),
            math=str(data["math"]),
            note=str(data.get("note", "") or ""),
        )


@dataclass
class Layer:
    id: str
    name: str
    kind: str
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    repeat: int | None = None
    children: list[Layer] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layer:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            kind=str(data.get("kind", "unknown")),
            inputs=[TensorSpec.from_dict(x) for x in data.get("inputs", [])],
            outputs=[TensorSpec.from_dict(x) for x in data.get("outputs", [])],
            ops=[Op.from_dict(x) for x in data.get("ops", [])],
            repeat=data.get("repeat"),
            children=[Layer.from_dict(c) for c in data.get("children", [])],
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class Architecture:
    name: str
    source: str = ""
    dims: dict[str, str] = field(default_factory=dict)
    blocks: list[Layer] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Architecture:
        blocks_raw = data.get("blocks") or data.get("layers") or []
        return cls(
            name=str(data.get("name", "model")),
            source=str(data.get("source", "") or ""),
            dims={str(k): str(v) for k, v in (data.get("dims") or {}).items()},
            blocks=[Layer.from_dict(b) for b in blocks_raw],
            notes=[str(n) for n in data.get("notes", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def iter_layers(self) -> Iterable[Layer]:
        def walk(nodes: list[Layer]) -> Iterable[Layer]:
            for node in nodes:
                yield node
                yield from walk(node.children)

        yield from walk(self.blocks)

    def first_leaf(self) -> Layer | None:
        for layer in self.iter_layers():
            if not layer.children:
                return layer
        return self.blocks[0] if self.blocks else None
