from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..ir import Architecture


def enrich_with_meta(
    arch: Architecture,
    model_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    max_header_files: int = 1,
) -> Architecture:
    """
    Attach tensor name → shape hints from the safetensors index and shard
    **headers only** (HTTP Range). Never loads tensor payloads into memory.
    """
    index = _try_load_index(model_id, revision=revision, token=token)
    shapes: dict[str, list[int]] = {}

    if index:
        weight_map: dict[str, str] = dict(index.get("weight_map") or {})
        arch.notes.append(
            f"Weight map entries: {len(weight_map)} (index only; shard bodies not downloaded)."
        )
        shard_files = sorted(set(weight_map.values()))[:max_header_files]
        for shard in shard_files:
            try:
                shapes.update(
                    _remote_safetensors_header_shapes(
                        model_id, shard, revision=revision, token=token
                    )
                )
            except Exception as exc:  # noqa: BLE001 — best-effort enrich
                arch.notes.append(f"Skipped shard header {shard}: {exc}")
    else:
        try:
            shapes.update(
                _remote_safetensors_header_shapes(
                    model_id, "model.safetensors", revision=revision, token=token
                )
            )
            arch.notes.append(
                "Read shapes from model.safetensors header only (Range request)."
            )
        except Exception as exc:  # noqa: BLE001
            arch.notes.append(f"No safetensors metadata available: {exc}")

    if shapes:
        sample = list(shapes.items())[:12]
        arch.notes.append(
            "Sample tensor shapes: " + ", ".join(f"{k}{list(v)}" for k, v in sample)
        )
        for layer in arch.iter_layers():
            prefix_hits = {
                k: v
                for k, v in shapes.items()
                if _name_hints_layer(k, layer.id, layer.name, layer.kind)
            }
            if prefix_hits:
                layer.meta["tensor_shapes"] = {
                    k: list(v) for k, v in list(prefix_hits.items())[:8]
                }

    return arch


def _try_load_index(
    model_id: str,
    *,
    revision: str | None,
    token: str | None,
) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --meta. "
            "Install with: pip install huggingface_hub"
        ) from exc

    path_local = Path(model_id)
    if path_local.is_dir():
        candidate = path_local / "model.safetensors.index.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        return None

    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            revision=revision,
            token=token,
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _remote_safetensors_header_shapes(
    repo_id: str,
    filename: str,
    *,
    revision: str | None,
    token: str | None,
) -> dict[str, list[int]]:
    """Fetch only the safetensors JSON header via HTTP Range."""
    local = Path(repo_id)
    if local.is_dir():
        return _read_safetensors_header_shapes(local / filename)

    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    # First request: 8-byte length prefix
    header_len = struct.unpack("<Q", _http_range(url, 0, 7, token=token))[0]
    if header_len > 64 * 1024 * 1024:
        raise ValueError(f"Safetensors header too large: {header_len} bytes")
    raw = _http_range(url, 8, 7 + header_len, token=token)
    header = json.loads(raw.decode("utf-8"))
    return _shapes_from_header(header)


def _http_range(url: str, start: int, end: int, *, token: str | None) -> bytes:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Range", f"bytes={start}-{end}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        # Some hosts ignore Range and return 200 with the full body — refuse that.
        if exc.code == 200:
            raise RuntimeError(
                "Server ignored Range request; refusing full-file download"
            ) from exc
        raise
    expected = end - start + 1
    if len(data) > expected * 4 and len(data) > 1024 * 1024:
        raise RuntimeError(
            f"Range response too large ({len(data)} bytes); refusing full download"
        )
    return data


def _read_safetensors_header_shapes(path: str | Path) -> dict[str, list[int]]:
    path = Path(path)
    with path.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return {}
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len > 64 * 1024 * 1024:
            raise ValueError(f"Safetensors header too large: {header_len} bytes")
        raw = f.read(header_len)
    return _shapes_from_header(json.loads(raw.decode("utf-8")))


def _shapes_from_header(header: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if isinstance(shape, list):
            shapes[key] = [int(x) for x in shape]
    return shapes


def _name_hints_layer(tensor_name: str, layer_id: str, layer_name: str, kind: str) -> bool:
    t = tensor_name.lower()
    if "embed" in layer_id or kind == "embedding":
        return "embed" in t
    if "lm_head" in layer_id or kind == "lm_head":
        return "lm_head" in t
    if kind in {"attention", "gqa_attention"}:
        return any(x in t for x in ("q_proj", "k_proj", "v_proj", "o_proj", "attn"))
    if kind in {"mlp", "swiglu_mlp", "moe"}:
        return any(x in t for x in ("mlp", "gate", "up_proj", "down_proj", "expert"))
    if kind in {"rmsnorm", "layernorm"}:
        return "norm" in t
    if "blocks" in layer_id:
        return "layers." in t or ".h." in t
    return False
