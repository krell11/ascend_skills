from __future__ import annotations

from typing import Any

from .ir import Op, TensorSpec


def _shapes(
    hidden: str = "H",
    seq: str = "S",
    batch: str = "B",
    heads: str = "N",
    head_dim: str = "D",
    vocab: str = "V",
    intermediate: str = "I",
    experts: str = "E",
) -> dict[str, str]:
    return {
        "B": batch,
        "S": seq,
        "H": hidden,
        "N": heads,
        "D": head_dim,
        "V": vocab,
        "I": intermediate,
        "E": experts,
    }


def ops_for_kind(kind: str, ctx: dict[str, Any] | None = None) -> list[Op]:
    """Return default ops for a layer kind. Unknown kinds get a placeholder."""
    ctx = ctx or {}
    s = _shapes(
        hidden=str(ctx.get("hidden", "H")),
        seq=str(ctx.get("seq", "S")),
        batch=str(ctx.get("batch", "B")),
        heads=str(ctx.get("heads", "N")),
        head_dim=str(ctx.get("head_dim", "D")),
        vocab=str(ctx.get("vocab", "V")),
        intermediate=str(ctx.get("intermediate", "I")),
        experts=str(ctx.get("experts", "E")),
    )
    B, S, H, N, D, V, I, E = (
        s["B"],
        s["S"],
        s["H"],
        s["N"],
        s["D"],
        s["V"],
        s["I"],
        s["E"],
    )

    catalog: dict[str, list[Op]] = {
        "embedding": [
            Op(
                "lookup",
                f"X[{B},{S},{H}] = Embed(token_ids[{B},{S}])",
                "Each token id picks one row from the embedding table. That row is the starting vector for the token.",
            ),
            Op(
                "scale",
                f"optional: X ← X · √{H}",
                "Some models scale embeddings by √H so early activations sit near unit variance.",
            ),
        ],
        "rope": [
            Op(
                "rotate",
                f"(Q', K') = RoPE(Q, K; θ)  with positions 0..{S}-1",
                "Rotary embeddings rotate pairs of dimensions by an angle that grows with position, so relative distance is baked into Q·K.",
            ),
        ],
        "rmsnorm": [
            Op(
                "rms",
                f"Y = X / RMS(X) · γ,  RMS(X) = √(mean(X²) + ε)",
                "Divide by the root-mean-square of the features (no mean subtraction), then rescale with learned γ. Stabilizes depth without centering.",
            ),
        ],
        "layernorm": [
            Op(
                "ln",
                f"Y = (X − μ) / √(σ² + ε) · γ + β",
                "Center and scale each token's features to zero mean and unit variance, then apply learned γ and β.",
            ),
        ],
        "attention": [
            Op(
                "project",
                f"Q,K,V = X W_qkv   →  [{B},{S},{N},{D}]",
                "One linear map splits the hidden state into queries, keys, and values for each attention head.",
            ),
            Op(
                "scores",
                f"A = softmax(Q Kᵀ / √{D} + mask)",
                "Dot-product similarity between every query and key, scaled by √D, masked (causal or padding), then turned into probabilities.",
            ),
            Op(
                "mix",
                f"O = A V",
                "Each query position becomes a weighted sum of value vectors, using the attention probabilities as weights.",
            ),
            Op(
                "out",
                f"Y = O W_o   →  [{B},{S},{H}]",
                "Concatenate heads and project back to the model hidden size H.",
            ),
        ],
        "gqa_attention": [
            Op(
                "project",
                f"Q=X W_q ({N} heads),  K,V=X W_kv (G groups)  →  [{B},{S},·,{D}]",
                "Queries keep many heads; keys and values share fewer groups. Each group is reused by several queries to save memory.",
            ),
            Op(
                "scores",
                f"A = softmax(Q Kᵀ / √{D} + mask)  (K,V repeated to {N})",
                "Same scaled-dot-product attention as MHA after repeating KV groups so each query head has a matching key/value.",
            ),
            Op(
                "mix",
                f"O = A V;  Y = O W_o",
                "Mix values with attention weights, then project the concatenated heads back to H.",
            ),
        ],
        "mlp": [
            Op(
                "up",
                f"U = X W_up   →  [{B},{S},{I}]",
                "Expand each token from H into a wider intermediate width I.",
            ),
            Op(
                "act",
                f"H = act(U)  (SiLU / GELU / …)",
                "Pointwise non-linearity lets the network bend the feature space between the two linear maps.",
            ),
            Op(
                "down",
                f"Y = H W_down   →  [{B},{S},{H}]",
                "Project the activated intermediate features back to hidden size H.",
            ),
        ],
        "swiglu_mlp": [
            Op(
                "gate_up",
                f"G, U = X W_gate, X W_up",
                "Two parallel projections: a gate branch G and a value branch U, both width I.",
            ),
            Op(
                "swiglu",
                f"H = SiLU(G) ⊙ U",
                "SiLU softens the gate; elementwise multiply keeps only the parts of U the gate allows through.",
            ),
            Op(
                "down",
                f"Y = H W_down   →  [{B},{S},{H}]",
                "Collapse the gated intermediate features back to H.",
            ),
        ],
        "moe": [
            Op(
                "route",
                f"p = softmax(X W_router)   →  [{B},{S},{E}]",
                "A small linear router scores every expert for each token and turns scores into probabilities.",
            ),
            Op(
                "dispatch",
                f"send tokens to top-k experts",
                "Only the k highest-scoring experts receive the token; the rest stay idle for that token.",
            ),
            Op(
                "expert",
                f"Y_e = MLP_e(X_e)  for experts e ∈ top-k",
                "Each chosen expert runs its own MLP on the tokens routed to it.",
            ),
            Op(
                "combine",
                f"Y = Σ_e p_e · Y_e",
                "Merge expert outputs with the router probabilities so the token returns one hidden vector.",
            ),
        ],
        "residual": [
            Op(
                "add",
                f"Y = X + F(X)",
                "Add the block's transform F(X) back onto the input so gradients and identity information can skip the transform.",
            ),
        ],
        "transformer_block": [
            Op(
                "attn_path",
                f"X ← X + Attn(Norm(X))",
                "Normalize, run attention, then residual-add. Tokens exchange information along the sequence.",
            ),
            Op(
                "mlp_path",
                f"X ← X + MLP(Norm(X))",
                "Normalize, run the feed-forward (or MoE) network, then residual-add. Features mix within each token.",
            ),
        ],
        "conv2d": [
            Op(
                "conv",
                f"Y = Conv2d(X; k×k, stride=s)  channels C_in → C_out",
                "A local k×k window slides over the spatial grid; each output channel is a learned filter response.",
            ),
        ],
        "vit_patch": [
            Op(
                "patch",
                f"P = unfold(image) → [{B}, N_patches, C·p·p]",
                "Cut the image into non-overlapping p×p patches and flatten each patch into a vector.",
            ),
            Op(
                "proj",
                f"X = P W_patch + pos_embed   →  [{B}, N_patches, {H}]",
                "Linearly map each patch into hidden size H and add a position embedding so order is known.",
            ),
        ],
        "cross_attention": [
            Op(
                "project",
                f"Q = X_q W_q;  K,V = X_kv W_kv",
                "Queries come from one stream (e.g. text); keys and values come from another (e.g. vision).",
            ),
            Op(
                "scores",
                f"A = softmax(Q Kᵀ / √{D})",
                "Each query position soft-selects which positions in the other stream to read.",
            ),
            Op(
                "mix",
                f"Y = (A V) W_o",
                "Read the selected values and project back to the query stream's width.",
            ),
        ],
        "lm_head": [
            Op(
                "proj",
                f"logits = X W_lm   →  [{B},{S},{V}]",
                "Map each hidden vector to a score over the vocabulary; higher logit means a more likely next token.",
            ),
        ],
        "dropout": [
            Op(
                "drop",
                f"Y = Dropout(X; p)",
                "During training, randomly zero a fraction p of activations to reduce co-adaptation.",
            ),
        ],
        "linear": [
            Op(
                "matvec",
                f"Y = X W + b",
                "Affine map: multiply by a weight matrix and optionally add a bias.",
            ),
        ],
    }

    if kind in catalog:
        return [Op(o.name, o.math, o.note) for o in catalog[kind]]
    return [
        Op(
            "forward",
            f"{kind}(X) → Y",
            "Generic forward map from input activations to output activations for this module kind.",
        )
    ]


def viz_kind_for(kind: str) -> str:
    """Interactive strip type used by the HTML widget."""
    mapping = {
        "attention": "attention",
        "gqa_attention": "attention",
        "cross_attention": "attention",
        "embedding": "embedding",
        "rmsnorm": "norm",
        "layernorm": "norm",
        "mlp": "mlp",
        "swiglu_mlp": "mlp",
        "moe": "moe",
        "residual": "residual",
        "transformer_block": "block",
        "lm_head": "lm_head",
        "rope": "rope",
        "vit_patch": "vit",
        "conv2d": "conv",
        "dropout": "dropout",
        "linear": "linear",
    }
    return mapping.get(kind, "generic")


def io_for_kind(
    kind: str, ctx: dict[str, Any] | None = None
) -> tuple[list[TensorSpec], list[TensorSpec]]:
    ctx = ctx or {}
    H = str(ctx.get("hidden", "H"))
    S = str(ctx.get("seq", "S"))
    B = str(ctx.get("batch", "B"))
    V = str(ctx.get("vocab", "V"))

    hidden = (
        [TensorSpec("X", [B, S, H])],
        [TensorSpec("Y", [B, S, H])],
    )
    special = {
        "embedding": (
            [TensorSpec("token_ids", [B, S])],
            [TensorSpec("X", [B, S, H])],
        ),
        "lm_head": (
            [TensorSpec("X", [B, S, H])],
            [TensorSpec("logits", [B, S, V])],
        ),
        "vit_patch": (
            [TensorSpec("image", [B, "C", "H_img", "W_img"])],
            [TensorSpec("X", [B, "N_patches", H])],
        ),
    }
    return special.get(kind, hidden)


KNOWN_KINDS = sorted(
    {
        "embedding",
        "rope",
        "rmsnorm",
        "layernorm",
        "attention",
        "gqa_attention",
        "mlp",
        "swiglu_mlp",
        "moe",
        "residual",
        "transformer_block",
        "conv2d",
        "vit_patch",
        "cross_attention",
        "lm_head",
        "dropout",
        "linear",
    }
)
