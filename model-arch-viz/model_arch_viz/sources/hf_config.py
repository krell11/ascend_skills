from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..formulas import ops_for_kind
from ..ir import Architecture, Layer, Op, TensorSpec
from .deepseek import architecture_from_deepseek
from .paper_ir import fill_missing_ops


def from_hf(
    model_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Architecture:
    """Fetch only config.json for ``model_id`` and build an IR."""
    cfg = _load_config_dict(model_id, revision=revision, token=token)
    return architecture_from_config(cfg, source=f"hf:{model_id}")


def from_config_file(path: str | Path) -> Architecture:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return architecture_from_config(cfg, source=str(path))


def architecture_from_config(
    cfg: dict[str, Any], *, source: str = "config.json"
) -> Architecture:
    cfg = _flatten_hf_config(cfg)
    built = architecture_from_deepseek(cfg, source=source)
    if built is not None:
        return built

    model_type = str(cfg.get("model_type") or cfg.get("architectures", ["model"])[0])
    name = str(
        cfg.get("_name_or_path")
        or (cfg.get("architectures") or [model_type])[0]
        or model_type
    )

    hidden = int(
        cfg.get("hidden_size")
        or cfg.get("d_model")
        or cfg.get("n_embd")
        or cfg.get("dim")
        or 0
    )
    layers = int(
        cfg.get("num_hidden_layers")
        or cfg.get("n_layer")
        or cfg.get("num_layers")
        or cfg.get("n_layers")
        or 0
    )
    heads = int(
        cfg.get("num_attention_heads")
        or cfg.get("n_head")
        or cfg.get("n_heads")
        or 0
    )
    kv_heads = int(
        cfg.get("num_key_value_heads") or cfg.get("num_kv_heads") or heads or 0
    )
    intermediate = int(
        cfg.get("intermediate_size")
        or cfg.get("ffn_dim")
        or cfg.get("n_inner")
        or (4 * hidden if hidden else 0)
        or 0
    )
    vocab = int(cfg.get("vocab_size") or 0)
    head_dim = int(cfg.get("head_dim") or (hidden // heads if heads else 0) or 0)
    num_experts = int(
        cfg.get("num_experts")
        or cfg.get("num_local_experts")
        or cfg.get("n_routed_experts")
        or 0
    )
    tie = bool(cfg.get("tie_word_embeddings", False))
    act = str(cfg.get("hidden_act") or cfg.get("activation_function") or "")

    dims: dict[str, str] = {
        "B": "batch",
        "S": "sequence length",
    }
    if hidden:
        dims["H"] = f"hidden size ({hidden})"
    if heads:
        dims["N"] = f"query heads ({heads})"
    if head_dim:
        dims["D"] = f"head dim ({head_dim})"
    if intermediate:
        dims["I"] = f"intermediate ({intermediate})"
    if vocab:
        dims["V"] = f"vocab ({vocab})"
    if num_experts:
        dims["E"] = f"experts ({num_experts})"

    ctx = {
        "hidden": "H",
        "seq": "S",
        "batch": "B",
        "heads": "N",
        "head_dim": "D",
        "vocab": "V",
        "intermediate": "I",
        "experts": "E",
    }

    use_gqa = bool(heads and kv_heads and kv_heads != heads)
    attn_kind = "gqa_attention" if use_gqa else "attention"
    mlp_kind = "swiglu_mlp" if "silu" in act.lower() or "swiglu" in act.lower() else "mlp"
    if num_experts:
        mlp_kind = "moe"

    blocks: list[Layer] = []

    if vocab and hidden:
        blocks.append(
            Layer(
                id="embed",
                name="Token Embedding",
                kind="embedding",
                inputs=[TensorSpec("token_ids", ["B", "S"])],
                outputs=[TensorSpec("X", ["B", "S", "H"])],
                ops=ops_for_kind("embedding", ctx),
                meta={"vocab_size": vocab, "hidden_size": hidden},
            )
        )

    if layers:
        children = [
            Layer(
                id="blocks.norm1",
                name="Pre-attention Norm",
                kind=_norm_kind(cfg),
                ops=ops_for_kind(_norm_kind(cfg), ctx),
            ),
            Layer(
                id="blocks.attn",
                name="Grouped-Query Attention" if use_gqa else "Self Attention",
                kind=attn_kind,
                ops=_attention_ops(ctx, use_gqa, heads, kv_heads, head_dim),
                meta={
                    "num_attention_heads": heads,
                    "num_key_value_heads": kv_heads,
                    "head_dim": head_dim,
                },
            ),
            Layer(
                id="blocks.norm2",
                name="Pre-MLP Norm",
                kind=_norm_kind(cfg),
                ops=ops_for_kind(_norm_kind(cfg), ctx),
            ),
            Layer(
                id="blocks.mlp",
                name="MoE FFN" if num_experts else ("SwiGLU MLP" if mlp_kind == "swiglu_mlp" else "MLP"),
                kind=mlp_kind,
                ops=ops_for_kind(mlp_kind, ctx),
                meta={"intermediate_size": intermediate, "num_experts": num_experts},
            ),
        ]
        blocks.append(
            Layer(
                id="blocks",
                name="TransformerBlock",
                kind="transformer_block",
                repeat=layers,
                inputs=[TensorSpec("X", ["B", "S", "H"])],
                outputs=[TensorSpec("X", ["B", "S", "H"])],
                ops=ops_for_kind("transformer_block", ctx),
                children=children,
                meta={"num_hidden_layers": layers},
            )
        )

    if hidden:
        blocks.append(
            Layer(
                id="final_norm",
                name="Final Norm",
                kind=_norm_kind(cfg),
                ops=ops_for_kind(_norm_kind(cfg), ctx),
            )
        )

    if vocab and hidden:
        blocks.append(
            Layer(
                id="lm_head",
                name="LM Head" + (" (tied)" if tie else ""),
                kind="lm_head",
                inputs=[TensorSpec("X", ["B", "S", "H"])],
                outputs=[TensorSpec("logits", ["B", "S", "V"])],
                ops=ops_for_kind("lm_head", ctx),
                meta={"tie_word_embeddings": tie},
            )
        )

    # Vision / multimodal hooks when config exposes them
    if _looks_like_vision(cfg):
        blocks.insert(
            0,
            Layer(
                id="vision",
                name="Vision / Patch Embed",
                kind="vit_patch",
                ops=ops_for_kind("vit_patch", ctx),
                meta=_vision_meta(cfg),
            ),
        )

    arch = Architecture(
        name=name,
        source=source,
        dims=dims,
        blocks=blocks,
        notes=_notes(cfg, model_type),
    )
    fill_missing_ops(arch)
    return arch


def _norm_kind(cfg: dict[str, Any]) -> str:
    rms = cfg.get("rms_norm_eps") is not None or str(cfg.get("model_type", "")).startswith(
        ("llama", "qwen", "mistral", "gemma")
    )
    return "rmsnorm" if rms else "layernorm"


def _attention_ops(
    ctx: dict[str, Any],
    use_gqa: bool,
    heads: int,
    kv_heads: int,
    head_dim: int,
) -> list[Op]:
    base = ops_for_kind("gqa_attention" if use_gqa else "attention", ctx)
    if use_gqa and heads and kv_heads:
        base = [
            Op(
                "project",
                f"Q=X W_q ({heads} heads); K,V=X W_kv ({kv_heads} groups) → [B,S,·,{head_dim or 'D'}]",
            ),
            Op("rope", "(Q', K') = RoPE(Q, K)  when position embeddings are rotary"),
            Op("scores", f"A = softmax(Q' K'ᵀ / √{head_dim or 'D'} + mask)"),
            Op("out", "Y = (A V) W_o"),
        ]
    return base


def _looks_like_vision(cfg: dict[str, Any]) -> bool:
    keys = set(cfg)
    return bool(
        keys
        & {
            "vision_config",
            "image_size",
            "patch_size",
            "mm_vision_tower",
            "visual",
        }
    ) or "vl" in str(cfg.get("model_type", "")).lower()


def _vision_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    vision = cfg.get("vision_config")
    if isinstance(vision, dict):
        return {
            "image_size": vision.get("image_size"),
            "patch_size": vision.get("patch_size"),
            "hidden_size": vision.get("hidden_size"),
        }
    return {
        "image_size": cfg.get("image_size"),
        "patch_size": cfg.get("patch_size"),
    }


def _notes(cfg: dict[str, Any], model_type: str) -> list[str]:
    notes = [
        f"Built from Hugging Face config only (model_type={model_type}).",
        "No weights were downloaded or loaded.",
    ]
    max_pos = cfg.get("max_position_embeddings") or cfg.get("n_positions")
    if max_pos:
        notes.append(f"max_position_embeddings={max_pos}")
    rope = cfg.get("rope_theta") or cfg.get("rotary_emb_base")
    if rope:
        notes.append(f"rope_theta={rope}")
    return notes


def _flatten_hf_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Lift a nested ``text_config`` up so language fields are readable.

    Multimodal configs keep the language stack under ``text_config`` and the
    tower under ``vision_config``. Top-level keys (architecture name, vision,
    quantization) win when both sides set the same field.
    """
    text = cfg.get("text_config")
    if not isinstance(text, dict):
        return cfg
    flat = dict(text)
    for key, value in cfg.items():
        if key == "text_config":
            continue
        flat[key] = value
    return flat


def _load_config_dict(
    model_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    path = Path(model_id)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if path.is_dir() and (path / "config.json").is_file():
        return json.loads((path / "config.json").read_text(encoding="utf-8"))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to fetch remote configs. "
            "Install with: pip install huggingface_hub"
        ) from exc

    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        revision=revision,
        token=token,
    )
    return json.loads(Path(config_path).read_text(encoding="utf-8"))
