from __future__ import annotations

from typing import Any

from ..formulas import ops_for_kind
from ..ir import Architecture, Layer, Op
from .paper_ir import fill_missing_ops

_NAMES = {
    "deepseek_v2": "DeepSeek-V2",
    "deepseek_v3": "DeepSeek-V3",
    "deepseek_v32": "DeepSeek-V3.2",
    "deepseek_v4": "DeepSeek-V4",
    "deepseek_v41": "DeepSeek-V4.1",
}


def architecture_from_deepseek(
    cfg: dict[str, Any], *, source: str = "config.json"
) -> Architecture | None:
    if not _is_deepseek(cfg):
        return None
    if _is_csa(cfg):
        arch = _csa_architecture(cfg, source)
    elif _is_mla(cfg):
        arch = _mla_architecture(cfg, source)
    else:
        return None
    fill_missing_ops(arch)
    return arch


def _is_deepseek(cfg: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(cfg.get("model_type") or ""),
            " ".join(str(a) for a in (cfg.get("architectures") or [])),
        ]
    ).lower()
    if "deepseek" in blob:
        return True
    return "compress_ratios" in cfg and "kv_source_layer_ids" in cfg


def _is_csa(cfg: dict[str, Any]) -> bool:
    model_type = str(cfg.get("model_type") or "").lower()
    if "deepseek_v4" in model_type:
        return True
    return "compress_ratios" in cfg or "kv_source_layer_ids" in cfg


def _is_mla(cfg: dict[str, Any]) -> bool:
    return _opt_int(cfg, "kv_lora_rank") is not None or _opt_int(cfg, "q_lora_rank") is not None


def _csa_architecture(cfg: dict[str, Any], source: str) -> Architecture:
    hidden = int(cfg.get("hidden_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or 0)
    ratios = [int(r) for r in (cfg.get("compress_ratios") or [])]
    notes = [
        "Built from the Hugging Face config only. No weights were downloaded.",
        f"model_type={cfg.get('model_type') or 'deepseek'}.",
    ]
    if len(ratios) < layers:
        notes.append(
            f"compress_ratios has {len(ratios)} entries and num_hidden_layers is {layers}. "
            "Missing backbone entries are treated as ratio 0."
        )
        ratios = ratios + [0] * (layers - len(ratios))
    backbone = ratios[:layers]
    draft_ratios = ratios[layers:]
    draft_n = len(draft_ratios)
    nextn = _opt_int(cfg, "num_nextn_predict_layers") or 0
    if draft_n == 0 and nextn:
        draft_n = nextn
        draft_ratios = [0] * draft_n
    if ratios:
        notes.append(_ratio_runs(ratios if len(ratios) >= layers else backbone + draft_ratios))

    kv = [i for i in (_ints(cfg.get("kv_source_layer_ids")) ) if 0 <= i < layers]
    index = [i for i in _ints(cfg.get("index_source_layer_ids")) if 0 <= i < layers]
    notes.append(
        "kv_source_layer_ids: "
        + _fmt_ids(kv)
        + ". index_source_layer_ids: "
        + _fmt_ids(index)
        + "."
    )
    split = _stack_split(cfg, layers, backbone)
    if split < layers:
        notes.append(
            f"The two stacks split at layer {split}"
            + (
                ", which is candidate_source_layer_id."
                if _opt_int(cfg, "candidate_source_layer_id") == split
                else ", where compress_ratios changes."
            )
        )
    _append_rope_note(notes, cfg)
    _append_quant_note(notes, cfg)

    blocks: list[Layer] = []
    vision = _vision_block(cfg, hidden)
    if vision is not None:
        blocks.append(vision)
    if hidden and _opt_int(cfg, "vocab_size"):
        blocks.append(_embedding(cfg, hidden))

    engram_ids = set(_ints(cfg.get("engram_layer_ids")))
    stacks: list[tuple[int, int, str, str]]
    if 0 < split < layers:
        stacks = [
            (0, split, "encoder", "Causal encoder"),
            (split, layers, "decoder", "Causal decoder"),
        ]
    else:
        stacks = [(0, layers, "backbone", "Backbone")]

    for start, end, sid, title in stacks:
        if start >= end:
            continue
        blocks.append(
            _csa_stack(cfg, start, end, sid, title, backbone, kv, index, engram_ids)
        )
        if sid == "encoder" and any(s[2] == "decoder" for s in stacks) and "hc_mult" in cfg:
            blocks.append(_mhc_block(cfg))

    if "hc_mult" in cfg and not any(b.kind == "mhc" for b in blocks):
        blocks.append(_mhc_block(cfg))

    blocks.append(_final_norm(cfg))
    if hidden and _opt_int(cfg, "vocab_size"):
        blocks.append(_lm_head(cfg, hidden))
    if draft_n:
        blocks.append(_dspark_block(cfg, layers, draft_n, draft_ratios))

    return Architecture(
        name=_display_name(cfg),
        source=source,
        dims=_csa_dims(cfg, hidden, layers, split),
        blocks=blocks,
        notes=notes,
    )


def _mla_architecture(cfg: dict[str, Any], source: str) -> Architecture:
    hidden = int(cfg.get("hidden_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or 0)
    dense_n = _opt_int(cfg, "first_k_dense_replace") or 0
    dense_n = max(0, min(dense_n, layers))
    moe_n = layers - dense_n
    notes = [
        "Built from the Hugging Face config only. No weights were downloaded.",
        f"model_type={cfg.get('model_type') or 'deepseek'}.",
    ]
    if dense_n:
        notes.append(
            f"first_k_dense_replace={dense_n}, so the first {dense_n} layers use a dense FFN "
            f"and the remaining {moe_n} use routed experts."
        )
    freq = cfg.get("moe_layer_freq")
    if freq not in (None, 1):
        notes.append(f"moe_layer_freq={freq}.")
    _append_rope_note(notes, cfg)
    _append_quant_note(notes, cfg)

    blocks: list[Layer] = []
    if hidden and _opt_int(cfg, "vocab_size"):
        blocks.append(_embedding(cfg, hidden))
    if dense_n:
        blocks.append(_mla_stack(cfg, "dense", "Dense layers", dense_n, moe=False))
    if moe_n:
        blocks.append(_mla_stack(cfg, "moe", "MoE layers", moe_n, moe=True))
    blocks.append(_final_norm(cfg))
    if hidden and _opt_int(cfg, "vocab_size"):
        blocks.append(_lm_head(cfg, hidden))
    nextn = _opt_int(cfg, "num_nextn_predict_layers") or 0
    if nextn:
        blocks.append(
            Layer(
                id="mtp",
                name="Multi-token prediction",
                kind="mtp",
                repeat=nextn,
                ops=_with_extra(
                    "mtp",
                    predict=f"num_nextn_predict_layers={nextn}.",
                ),
                meta={"num_nextn_predict_layers": nextn},
            )
        )
    return Architecture(
        name=_display_name(cfg),
        source=source,
        dims=_mla_dims(cfg, hidden, layers, dense_n, moe_n),
        blocks=blocks,
        notes=notes,
    )


def _csa_stack(
    cfg: dict[str, Any],
    start: int,
    end: int,
    sid: str,
    title: str,
    ratios: list[int],
    kv: list[int],
    index: list[int],
    engram_ids: set[int],
) -> Layer:
    children: list[Layer] = []
    for kind, lo, hi in _groups(start, end, ratios, set(kv), set(index), engram_ids):
        if kind == "engram":
            children.append(_engram_block(cfg, lo))
            continue
        ratio = ratios[lo]
        mode = _mode(lo, ratio, set(kv), set(index))
        children.append(_csa_block(cfg, sid, lo, hi, mode, ratio))
    if _opt_int(cfg, "n_routed_experts"):
        children.append(
            _moe_block(
                f"{sid}.moe",
                f"{title} MoE",
                cfg,
                routed=_opt_int(cfg, "n_routed_experts") or 0,
                topk=_opt_int(cfg, "num_experts_per_tok") or 0,
                shared=_opt_int(cfg, "n_shared_experts"),
                include_width=True,
            )
        )
    kv_here = [i for i in kv if start <= i < end]
    index_here = [i for i in index if start <= i < end]
    return Layer(
        id=sid,
        name=title,
        kind="transformer_block",
        repeat=end - start,
        ops=[
            Op(
                "role",
                f"layers {start}–{end - 1}",
                (
                    f"This stack is layers {start}–{end - 1} of num_hidden_layers. "
                    f"KV is built at {_fmt_ids(kv_here)}. "
                    f"The indexer runs at {_fmt_ids(index_here)}. "
                    "Every other layer with a positive compress ratio reuses those."
                ),
            )
        ],
        children=children,
        meta={"layer_start": start, "layer_end": end - 1},
    )


def _groups(
    start: int,
    end: int,
    ratios: list[int],
    kv: set[int],
    index: set[int],
    engram_ids: set[int],
):
    """Yield ('engram', layer, layer+1) or ('attn', lo, hi) covering [start, end)."""
    i = start
    while i < end:
        if i in engram_ids:
            yield ("engram", i, i + 1)
        mode = _mode(i, ratios[i], kv, index)
        ratio = ratios[i]
        j = i + 1
        while (
            j < end
            and j not in engram_ids
            and ratios[j] == ratio
            and _mode(j, ratios[j], kv, index) == mode
        ):
            j += 1
        yield ("attn", i, j)
        i = j


def _csa_block(
    cfg: dict[str, Any],
    sid: str,
    lo: int,
    hi: int,
    mode: str,
    ratio: int,
) -> Layer:
    labels = {
        "swa": "Sliding window",
        "full": f"CSA2 full, ratio {ratio}",
        "reuse": f"CSA2 reuse, ratio {ratio}",
        "reindex": "CSA2 reindex",
    }
    span = _span(lo, hi)
    candidate = _opt_int(cfg, "candidate_source_layer_id")
    is_candidate = candidate is not None and lo <= candidate < hi and mode == "full"
    meta: dict[str, Any] = {
        "compress_ratio": ratio,
        "mode": mode,
        "layer_start": lo,
        "layer_end": hi - 1,
    }
    for key in (
        "sliding_window",
        "index_topk",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "head_dim",
        "qk_rope_head_dim",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
    ):
        if cfg.get(key) is not None:
            meta[key] = cfg[key]
    if is_candidate:
        for key in ("candidate_source_layer_id", "candidate_topk_blocks", "candidate_block_size"):
            if cfg.get(key) is not None:
                meta[key] = cfg[key]
    return Layer(
        id=f"{sid}.attn.{lo}",
        name=f"{labels[mode]}, {span}",
        kind="csa2",
        repeat=(hi - lo) if hi - lo > 1 else None,
        ops=_csa_ops(cfg, mode, ratio, lo, hi, is_candidate),
        meta=meta,
    )


def _csa_ops(
    cfg: dict[str, Any],
    mode: str,
    ratio: int,
    lo: int,
    hi: int,
    is_candidate: bool,
) -> list[Op]:
    catalog = {op.name: op for op in ops_for_kind("csa2")}
    if mode == "swa" or ratio == 0:
        names = ["query", "window", "attend"]
    elif mode == "reuse":
        names = ["query", "window", "attend"]
    elif mode == "reindex":
        names = ["query", "window", "index", "attend"]
    else:
        names = ["query", "window", "compress", "index", "attend"]
    span = _span(lo, hi)
    query_bits = _present(
        cfg,
        [
            ("q_lora_rank", "q_lora_rank"),
            ("num_attention_heads", "heads"),
            ("num_key_value_heads", "KV heads"),
            ("head_dim", "head_dim"),
            ("qk_rope_head_dim", "qk_rope_head_dim"),
        ],
    )
    window_bits = [f"Applies to {span}."]
    if cfg.get("sliding_window") is not None:
        window_bits.append(f"sliding_window={cfg['sliding_window']}.")
    if ratio == 0:
        window_bits.append("compress_ratios is 0, so there is no global branch.")
    compress_bits = [f"compress_ratios={ratio} on {span}."]
    if mode == "full":
        compress_bits.append("This layer is in kv_source_layer_ids, so it builds the compressed KV.")
    if cfg.get("compress_rope_theta") is not None and ratio > 0:
        compress_bits.append(f"compress_rope_theta={cfg['compress_rope_theta']}.")
    index_bits = []
    if cfg.get("index_topk") is not None:
        index_bits.append(f"index_topk={cfg['index_topk']}.")
    if cfg.get("index_n_heads") is not None:
        index_bits.append(
            f"index_n_heads={cfg['index_n_heads']}, index_head_dim={cfg.get('index_head_dim')}."
        )
    if mode == "full":
        index_bits.append("This layer is in index_source_layer_ids and publishes the top indices.")
    elif mode == "reindex":
        index_bits.append(
            "This layer is in index_source_layer_ids but not kv_source_layer_ids, so it rescores a reused key cache."
        )
    if is_candidate:
        index_bits.append(
            "candidate_source_layer_id="
            f"{cfg.get('candidate_source_layer_id')}, "
            f"candidate_topk_blocks={cfg.get('candidate_topk_blocks')}, "
            f"candidate_block_size={cfg.get('candidate_block_size')}."
        )
    attend_bits = []
    if mode == "reuse":
        attend_bits.append(
            "This layer is in neither source list, so it reuses the previous compressed KV and indices. The sliding window is still its own."
        )
    if cfg.get("o_groups") is not None:
        attend_bits.append(f"o_groups={cfg['o_groups']}, o_lora_rank={cfg.get('o_lora_rank')}.")
    extra = {
        "query": " ".join(query_bits),
        "window": " ".join(window_bits),
        "compress": " ".join(compress_bits),
        "index": " ".join(index_bits),
        "attend": " ".join(attend_bits),
    }
    ops: list[Op] = []
    for name in names:
        base = catalog[name]
        suffix = extra.get(name, "").strip()
        note = base.note if not suffix else f"{base.note} {suffix}"
        ops.append(Op(name, base.math, note))
    return ops


def _engram_block(cfg: dict[str, Any], layer_id: int) -> Layer:
    ids = _ints(cfg.get("engram_layer_ids"))
    sizes = list(cfg.get("engram_num_embeddings") or [])
    size = None
    if layer_id in ids:
        pos = ids.index(layer_id)
        if pos < len(sizes):
            size = sizes[pos]
    meta: dict[str, Any] = {"layer_id": layer_id}
    for key in (
        "engram_vocab_size",
        "engram_n_heads",
        "engram_head_dim",
        "engram_max_ngram_size",
        "engram_compressed_vocab_size",
        "engram_pad_token_id",
    ):
        if cfg.get(key) is not None:
            meta[key] = cfg[key]
    if size is not None:
        meta["engram_num_embeddings"] = size
    bits = [f"engram_layer_ids includes layer {layer_id}."]
    if size is not None:
        bits.append(f"engram_num_embeddings for this module is {size}.")
    if cfg.get("engram_vocab_size") is not None:
        bits.append(f"engram_vocab_size={cfg['engram_vocab_size']}.")
    if cfg.get("engram_compressed_vocab_size") is not None:
        bits.append(f"engram_compressed_vocab_size={cfg['engram_compressed_vocab_size']}.")
    return Layer(
        id=f"engram.{layer_id}",
        name=f"Engram at layer {layer_id}",
        kind="engram",
        ops=_with_extra("engram", hash=" ".join(bits), gate=" ".join(bits)),
        meta=meta,
    )


def _mhc_block(cfg: dict[str, Any]) -> Layer:
    meta = {
        key: cfg[key]
        for key in ("hc_mult", "hc_sinkhorn_iters", "hc_eps")
        if key in cfg and cfg[key] is not None
    }
    bits = [f"{key}={cfg[key]}" for key in ("hc_mult", "hc_sinkhorn_iters", "hc_eps") if cfg.get(key) is not None]
    return Layer(
        id="mhc",
        name="Hyper-connections",
        kind="mhc",
        ops=_with_extra(
            "mhc",
            mix="Config values: " + ", ".join(bits) + ". The same mix is used at each attention and MoE site."
            if bits
            else "",
        ),
        meta=meta,
    )


def _dspark_block(
    cfg: dict[str, Any], start: int, count: int, ratios: list[int]
) -> Layer:
    meta: dict[str, Any] = {}
    for key in (
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
        "dspark_n_routed_experts",
        "dspark_num_experts_per_tok",
        "num_nextn_predict_layers",
    ):
        if cfg.get(key) is not None:
            meta[key] = cfg[key]
    end = start + count - 1
    bits = [f"Draft layers {start}–{end}."]
    if cfg.get("dspark_block_size") is not None:
        bits.append(f"dspark_block_size={cfg['dspark_block_size']}.")
    if cfg.get("dspark_markov_rank") is not None:
        bits.append(f"dspark_markov_rank={cfg['dspark_markov_rank']}.")
    if cfg.get("dspark_noise_token_id") is not None:
        bits.append(f"dspark_noise_token_id={cfg['dspark_noise_token_id']}.")
    if cfg.get("dspark_target_layer_ids") is not None:
        bits.append(f"dspark_target_layer_ids={_fmt_ids(_ints(cfg.get('dspark_target_layer_ids')))}.")
    if ratios:
        bits.append("Draft compress_ratios: " + ", ".join(str(r) for r in ratios) + ".")
    children: list[Layer] = []
    if ratios and all(int(r) == 0 for r in ratios):
        children.append(
            _csa_block(cfg, "dspark", start, start + count, "swa", 0)
        )
    elif ratios:
        children.append(
            Layer(
                id="dspark.ratios",
                name="Draft compress ratios",
                kind="csa2",
                ops=[
                    Op(
                        "window",
                        "draft compress_ratios = [" + ", ".join(str(r) for r in ratios) + "]",
                        "These entries sit after the backbone in compress_ratios.",
                    )
                ],
                meta={"compress_ratio": int(ratios[0]), "mode": "swa" if int(ratios[0]) == 0 else "full"},
            )
        )
    routed = _opt_int(cfg, "dspark_n_routed_experts")
    topk = _opt_int(cfg, "dspark_num_experts_per_tok")
    if routed and topk:
        draft_cfg = {
            "scoring_func": cfg.get("scoring_func"),
            "topk_method": cfg.get("topk_method"),
            "norm_topk_prob": cfg.get("norm_topk_prob"),
        }
        children.append(
            _moe_block(
                "dspark.moe",
                "Draft MoE",
                draft_cfg,
                routed=routed,
                topk=topk,
                shared=None,
                include_width=False,
            )
        )
    return Layer(
        id="dspark",
        name="DSpark draft",
        kind="dspark",
        repeat=count,
        ops=_with_extra("dspark", draft=" ".join(bits), verify=" ".join(bits)),
        children=children,
        meta=meta,
    )


def _mla_stack(cfg: dict[str, Any], sid: str, title: str, repeat: int, *, moe: bool) -> Layer:
    mla_meta = {
        key: cfg[key]
        for key in (
            "hidden_size",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "v_head_dim",
        )
        if cfg.get(key) is not None
    }
    mla_bits = _present(
        cfg,
        [
            ("q_lora_rank", "q_lora_rank"),
            ("kv_lora_rank", "kv_lora_rank"),
            ("qk_nope_head_dim", "qk_nope_head_dim"),
            ("qk_rope_head_dim", "qk_rope_head_dim"),
            ("num_attention_heads", "heads"),
        ],
    )
    children = [
        _norm_layer(cfg, f"{sid}.norm1", "Pre-attention norm"),
        Layer(
            id=f"{sid}.mla",
            name="Multi-head latent attention",
            kind="mla",
            ops=_with_extra("mla", q_down=" ".join(mla_bits), kv_down=" ".join(mla_bits)),
            meta=mla_meta,
        ),
        _norm_layer(cfg, f"{sid}.norm2", "Pre-FFN norm"),
    ]
    if moe:
        children.append(
            _moe_block(
                f"{sid}.moe",
                "DeepSeekMoE",
                cfg,
                routed=_opt_int(cfg, "n_routed_experts") or _opt_int(cfg, "num_experts") or 0,
                topk=_opt_int(cfg, "num_experts_per_tok") or 0,
                shared=_opt_int(cfg, "n_shared_experts"),
                include_width=True,
            )
        )
    else:
        inter = _opt_int(cfg, "intermediate_size")
        children.append(
            Layer(
                id=f"{sid}.ffn",
                name="SwiGLU",
                kind="swiglu_mlp",
                ops=_with_extra(
                    "swiglu_mlp",
                    gate_up=f"intermediate_size={inter}." if inter else "",
                ),
                meta={"intermediate_size": inter} if inter else {},
            )
        )
    return Layer(
        id=sid,
        name=title,
        kind="transformer_block",
        repeat=repeat,
        ops=_with_extra(
            "transformer_block",
            attn_path="Attention is multi-head latent attention.",
            mlp_path="The feed-forward is routed experts." if moe else "The feed-forward is a dense SwiGLU.",
        ),
        children=children,
    )


def _moe_block(
    layer_id: str,
    name: str,
    cfg: dict[str, Any],
    *,
    routed: int,
    topk: int,
    shared: int | None,
    include_width: bool,
) -> Layer:
    scoring = str(cfg.get("scoring_func") or "sigmoid")
    no_groups = "n_group" not in cfg and "topk_group" not in cfg
    meta: dict[str, Any] = {
        "n_routed_experts": routed,
        "num_experts_per_tok": topk,
        "scoring_func": scoring,
    }
    if shared is not None:
        meta["n_shared_experts"] = shared
    if "n_group" in cfg and cfg.get("n_group") is not None:
        meta["n_group"] = int(cfg["n_group"])
    elif no_groups:
        meta["n_group"] = 1
        meta["topk_group"] = 1
    if cfg.get("topk_group") is not None:
        meta["topk_group"] = int(cfg["topk_group"])
    if cfg.get("routed_scaling_factor") is not None:
        meta["routed_scaling_factor"] = cfg["routed_scaling_factor"]
    if include_width and cfg.get("moe_intermediate_size") is not None:
        meta["moe_intermediate_size"] = cfg["moe_intermediate_size"]

    ops: list[Op] = []
    for op in ops_for_kind("deepseek_moe"):
        if op.name == "affinity" and scoring != "sigmoid":
            ops.append(
                Op(
                    "affinity",
                    f"s_i = {scoring}(u · e_i)",
                    f"scoring_func is {scoring}. Each expert is scored on its own, not with a softmax over the pool.",
                )
            )
        elif op.name == "groups" and no_groups:
            ops.append(
                Op(
                    "groups",
                    "top-k over the full routed pool",
                    "This config has no n_group, so routing is not split into expert groups.",
                )
            )
        elif op.name == "balance":
            ops.append(
                Op(
                    op.name,
                    op.math,
                    op.note
                    + f" num_experts_per_tok={topk} out of n_routed_experts={routed}.",
                )
            )
        elif op.name == "experts":
            bits = []
            if shared is None:
                note = "The config gives the routed-expert count and does not give a shared-expert count for this module."
            else:
                note = op.note
                bits.append(f"n_shared_experts={shared}.")
            if include_width and cfg.get("moe_intermediate_size") is not None:
                bits.append(f"moe_intermediate_size={cfg['moe_intermediate_size']}.")
            if cfg.get("swiglu_limit") is not None:
                bits.append(f"swiglu_limit={cfg['swiglu_limit']}.")
            ops.append(Op(op.name, op.math, _join(note, bits)))
        elif op.name == "combine":
            bits = []
            if cfg.get("routed_scaling_factor") is not None:
                bits.append(f"routed_scaling_factor={cfg['routed_scaling_factor']}.")
            if cfg.get("norm_topk_prob"):
                bits.append("norm_topk_prob is true, so the selected scores are renormalized.")
            if cfg.get("topk_method"):
                bits.append(f"topk_method={cfg['topk_method']}.")
            ops.append(Op(op.name, op.math, _join(op.note, bits)))
        else:
            ops.append(Op(op.name, op.math, op.note))
    return Layer(id=layer_id, name=name, kind="deepseek_moe", ops=ops, meta=meta)


def _vision_block(cfg: dict[str, Any], language_hidden: int) -> Layer | None:
    vision = cfg.get("vision_config")
    if not isinstance(vision, dict):
        return None
    layers = int(vision.get("num_hidden_layers") or 0)
    hidden = vision.get("hidden_size")
    heads = vision.get("num_attention_heads")
    intermediate = vision.get("intermediate_size")
    patch = vision.get("patch_size")
    down = vision.get("downsample_ratio")
    children: list[Layer] = []
    if patch is not None or hidden is not None:
        children.append(
            Layer(
                id="vision.patch",
                name="Patch projection",
                kind="linear",
                ops=[
                    Op(
                        "proj",
                        f"patch {patch}×{patch} → width {hidden}",
                        "patch_size and hidden_size come from vision_config.",
                    )
                ],
            )
        )
    if layers:
        children.append(
            Layer(
                id="vision.block",
                name="Vision block",
                kind="transformer_block",
                repeat=layers,
                ops=[
                    Op(
                        "block",
                        f"{layers} blocks, {heads} heads, feed-forward width {intermediate}",
                        "Counts come from vision_config. The config does not name the vision norm or activation.",
                    )
                ],
                meta={
                    "num_hidden_layers": layers,
                    "hidden_size": hidden,
                    "num_attention_heads": heads,
                    "intermediate_size": intermediate,
                },
            )
        )
    if down:
        children.append(
            Layer(
                id="vision.unshuffle",
                name="Pixel unshuffle",
                kind="linear",
                ops=[
                    Op(
                        "unshuffle",
                        f"{down}×{down} spatial neighborhood folds into channels",
                        "downsample_ratio comes from vision_config.",
                    )
                ],
            )
        )
    proj_bits = [f"Language hidden size is {language_hidden}."]
    if cfg.get("image_token_id") is not None:
        proj_bits.append(f"image_token_id={cfg['image_token_id']}.")
    if vision.get("max_image_tokens") is not None:
        proj_bits.append(f"max_image_tokens={vision['max_image_tokens']}.")
    children.append(
        Layer(
            id="vision.proj",
            name="Vision projector",
            kind="mlp",
            ops=[
                Op(
                    "proj",
                    f"vision features → hidden {language_hidden}",
                    " ".join(proj_bits),
                )
            ],
        )
    )
    return Layer(
        id="vision",
        name="Vision encoder",
        kind="vit_patch",
        ops=[
            Op(
                "path",
                "image → vision blocks → projector → language hidden",
                "The tower is vision_config. Its tokens are spliced into the text sequence.",
            )
        ],
        children=children,
        meta={
            "patch_size": patch,
            "hidden_size": hidden,
            "downsample_ratio": down,
            "max_image_tokens": vision.get("max_image_tokens"),
        },
    )


def _embedding(cfg: dict[str, Any], hidden: int) -> Layer:
    vocab = int(cfg.get("vocab_size") or 0)
    bits = [f"vocab_size={vocab}, hidden_size={hidden}."]
    if cfg.get("image_token_id") is not None:
        bits.append(
            f"Where the id is image_token_id {cfg['image_token_id']}, the vision projector vector is used."
        )
    tie = bool(cfg.get("tie_word_embeddings", False))
    if tie:
        bits.append("tie_word_embeddings is true.")
    return Layer(
        id="embed",
        name="Token embedding",
        kind="embedding",
        ops=[op for op in _with_extra("embedding", lookup=" ".join(bits)) if op.name == "lookup"],
        meta={"vocab_size": vocab, "hidden_size": hidden},
    )


def _final_norm(cfg: dict[str, Any]) -> Layer:
    return _norm_layer(cfg, "final_norm", "Final norm")


def _norm_layer(cfg: dict[str, Any], layer_id: str, name: str) -> Layer:
    eps = cfg.get("rms_norm_eps")
    if eps is None:
        return Layer(id=layer_id, name=name, kind="layernorm", ops=ops_for_kind("layernorm"))
    return Layer(
        id=layer_id,
        name=name,
        kind="rmsnorm",
        ops=[
            Op(
                "rms",
                f"Y = X / RMS(X) · γ,  ε = {eps}",
                "rms_norm_eps comes from the text config.",
            )
        ],
        meta={"rms_norm_eps": eps},
    )


def _lm_head(cfg: dict[str, Any], hidden: int) -> Layer:
    vocab = int(cfg.get("vocab_size") or 0)
    tie = bool(cfg.get("tie_word_embeddings", False))
    return Layer(
        id="lm_head",
        name="LM head" + (" (tied)" if tie else ""),
        kind="lm_head",
        ops=_with_extra(
            "lm_head",
            proj=f"hidden {hidden} → vocab {vocab}."
            + (" The head is tied to the input embedding." if tie else " tie_word_embeddings is false."),
        ),
        meta={"tie_word_embeddings": tie, "vocab_size": vocab, "hidden_size": hidden},
    )


def _stack_split(cfg: dict[str, Any], layers: int, ratios: list[int]) -> int:
    candidate = _opt_int(cfg, "candidate_source_layer_id")
    if candidate is not None and 0 < candidate < layers:
        return candidate
    for i in range(1, len(ratios)):
        if ratios[i] != ratios[i - 1] and ratios[i] == 1 and ratios[i - 1] > 1:
            return i
    return layers


def _mode(layer_id: int, ratio: int, kv: set[int], index: set[int]) -> str:
    if int(ratio) == 0:
        return "swa"
    if layer_id in kv:
        return "full"
    if layer_id in index:
        return "reindex"
    return "reuse"


def _display_name(cfg: dict[str, Any]) -> str:
    model_type = str(cfg.get("model_type") or "")
    if model_type in _NAMES:
        return _NAMES[model_type]
    archs = cfg.get("architectures") or []
    if archs:
        return str(archs[0])
    return model_type or "DeepSeek"


def _csa_dims(cfg: dict[str, Any], hidden: int, layers: int, split: int) -> dict[str, str]:
    dims: dict[str, str] = {}
    if hidden:
        dims["H"] = f"{hidden} hidden"
    if layers:
        if 0 < split < layers:
            dims["L"] = f"{layers} layers ({split} + {layers - split})"
        else:
            dims["L"] = f"{layers} layers"
    heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads")
    if heads:
        dims["N"] = f"{heads} query heads" + (f", {kv_heads} KV heads" if kv_heads else "")
    if cfg.get("head_dim"):
        rope = cfg.get("qk_rope_head_dim")
        dims["D"] = f"{cfg['head_dim']} head dim" + (f" ({rope} RoPE)" if rope else "")
    if cfg.get("q_lora_rank"):
        dims["d_q"] = f"{cfg['q_lora_rank']} query latent"
    routed = cfg.get("n_routed_experts")
    topk = cfg.get("num_experts_per_tok")
    if routed:
        dims["E"] = f"{routed} routed experts" + (f", top {topk}" if topk else "")
    if cfg.get("sliding_window"):
        dims["W"] = f"sliding window {cfg['sliding_window']}"
    if cfg.get("vocab_size"):
        dims["V"] = f"{cfg['vocab_size']} vocab"
    if cfg.get("max_position_embeddings"):
        dims["ctx"] = f"{cfg['max_position_embeddings']} positions"
    return dims


def _mla_dims(
    cfg: dict[str, Any], hidden: int, layers: int, dense_n: int, moe_n: int
) -> dict[str, str]:
    dims: dict[str, str] = {}
    if hidden:
        dims["H"] = f"{hidden} hidden"
    if layers:
        if dense_n and moe_n:
            dims["L"] = f"{layers} layers ({dense_n} dense + {moe_n} MoE)"
        else:
            dims["L"] = f"{layers} layers"
    if cfg.get("num_attention_heads"):
        dims["N"] = f"{cfg['num_attention_heads']} heads"
    if cfg.get("kv_lora_rank"):
        dims["d_c"] = f"{cfg['kv_lora_rank']} KV latent"
    if cfg.get("q_lora_rank"):
        dims["d_q"] = f"{cfg['q_lora_rank']} query latent"
    if cfg.get("qk_nope_head_dim"):
        dims["d_h"] = f"{cfg['qk_nope_head_dim']} content head"
    if cfg.get("qk_rope_head_dim"):
        dims["d_r"] = f"{cfg['qk_rope_head_dim']} RoPE head"
    if cfg.get("n_routed_experts"):
        dims["E"] = f"{cfg['n_routed_experts']} routed experts"
    if cfg.get("num_experts_per_tok"):
        dims["K"] = f"{cfg['num_experts_per_tok']} experts / token"
    if cfg.get("vocab_size"):
        dims["V"] = f"{cfg['vocab_size']} vocab"
    return dims


def _append_rope_note(notes: list[str], cfg: dict[str, Any]) -> None:
    bits = []
    if cfg.get("rope_theta") is not None:
        bits.append(f"rope_theta={cfg['rope_theta']}")
    if cfg.get("compress_rope_theta") is not None:
        bits.append(f"compress_rope_theta={cfg['compress_rope_theta']}")
    if cfg.get("max_position_embeddings") is not None:
        bits.append(f"max_position_embeddings={cfg['max_position_embeddings']}")
    scaling = cfg.get("rope_scaling")
    if isinstance(scaling, dict) and scaling:
        kind = scaling.get("rope_type") or scaling.get("type") or "scaling"
        factor = scaling.get("factor")
        original = scaling.get("original_max_position_embeddings")
        piece = str(kind)
        if factor is not None:
            piece += f" factor {factor}"
        if original is not None:
            piece += f" from {original}"
        bits.append(piece)
    if bits:
        notes.append("RoPE: " + ", ".join(bits) + ".")


def _append_quant_note(notes: list[str], cfg: dict[str, Any]) -> None:
    quant = cfg.get("quantization_config")
    if not isinstance(quant, dict) or not quant:
        return
    bits = []
    for key in ("quant_method", "activation_scheme", "weight_block_size", "scale_fmt", "expert_dtype"):
        if quant.get(key) is not None:
            bits.append(f"{key}={quant[key]}")
    if bits:
        notes.append("Quantization from the config: " + ", ".join(bits) + ".")


def _ratio_runs(ratios: list[int]) -> str:
    parts: list[str] = []
    start = 0
    for i in range(1, len(ratios) + 1):
        if i == len(ratios) or ratios[i] != ratios[start]:
            parts.append(f"ratio {ratios[start]} on {_span(start, i)}")
            start = i
    return "compress_ratios: " + "; ".join(parts) + "."


def _span(lo: int, hi: int) -> str:
    if hi - lo <= 1:
        return f"layer {lo}"
    return f"layers {lo}–{hi - 1}"


def _fmt_ids(ids: list[int]) -> str:
    return ", ".join(str(i) for i in ids) if ids else "none"


def _ints(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _opt_int(cfg: dict[str, Any], key: str) -> int | None:
    value = cfg.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _present(cfg: dict[str, Any], pairs: list[tuple[str, str]]) -> list[str]:
    return [f"{label}={cfg[key]}" for key, label in pairs if cfg.get(key) is not None]


def _with_extra(kind: str, **extra: str) -> list[Op]:
    ops: list[Op] = []
    for op in ops_for_kind(kind):
        suffix = extra.get(op.name, "").strip()
        note = op.note if not suffix else f"{op.note} {suffix}"
        ops.append(Op(op.name, op.math, note))
    return ops


def _join(note: str, bits: list[str]) -> str:
    extra = " ".join(bit for bit in bits if bit)
    return note if not extra else f"{note} {extra}"
