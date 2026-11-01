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
        "mla": [
            Op(
                "q_down",
                f"c_Q = RMSNorm(h W_DQ)    [{H}] → [d_q]",
                "Compress the query through a low-rank down projection, then normalize the latent. Training stores a narrow activation instead of a full per-head query.",
            ),
            Op(
                "q_up",
                "q^C = c_Q W_UQ,   q^R = RoPE(c_Q W_QR),   q = [q^C ; q^R]",
                "Split the query latent into a content part and a rotary part. Only the rotary part is rotated with position.",
            ),
            Op(
                "kv_down",
                f"c_KV = RMSNorm(h W_DKV)    [{H}] → [d_c]",
                "Keys and values share one low-rank latent. Inference caches this vector instead of a full per-head key and value.",
            ),
            Op(
                "kv_up",
                "k^C = c_KV W_UK,   v^C = c_KV W_UV",
                "Up-projections recover per-head content keys and values. At inference those matrices can be absorbed into the query up-projection and the output projection.",
            ),
            Op(
                "rope_key",
                "k^R = RoPE(h W_KR),   k = [k^C ; k^R]",
                "A small extra key carries position with RoPE and is shared by every head. It is cached next to the KV latent.",
            ),
            Op(
                "scores",
                "o_i = Σ_{j≤t} softmax(q_i · k_j / √(d_h + d_r)) v^C_j",
                "Causal scaled dot-product attention. Position enters only through the rotary slices. The mixed vector is the content value.",
            ),
            Op(
                "out",
                f"u = W_O [o_1 ; … ; o_N]    → [{H}]",
                "Concatenate heads and project back to the hidden size.",
            ),
        ],
        "deepseek_moe": [
            Op(
                "affinity",
                "s_i = sigmoid(u · e_i)",
                "Each routed expert scores the token on its own with a sigmoid. The scores are not a softmax over the whole expert pool.",
            ),
            Op(
                "balance",
                "choose top-k by (s_i + b_i); mix with s_i only",
                "A per-expert bias is added only while picking the top-k. The mix weight stays the original affinity, so load balance does not rewrite the gate.",
            ),
            Op(
                "groups",
                "keep the top groups, then the top-k experts inside them",
                "Routed experts are split into groups. A token may enter only a few groups, which caps how many devices it talks to.",
            ),
            Op(
                "experts",
                "y_shared = SwiGLU_shared(u),   y_i = SwiGLU_i(u) for selected i",
                "Shared experts always run. Each chosen routed expert is its own small SwiGLU.",
            ),
            Op(
                "combine",
                "y = y_shared + α · Σ g_i y_i,   g = renorm(s) over the selected experts",
                "Add the shared expert to a scaled mixture of the routed experts. Selected sigmoid scores are normalized so the gates sum to one.",
            ),
        ],
        "mtp": [
            Op(
                "combine",
                f"h' = M [ RMSNorm(h_t) ; RMSNorm(Embed(token_{{t+1}})) ]",
                "Concatenate the main-model state at this position with the embedding of the next token, then project back to the hidden size.",
            ),
            Op(
                "block",
                "h_mtp = Transformer(h')",
                "One more transformer block keeps a causal chain at this extra prediction depth.",
            ),
            Op(
                "predict",
                f"P(token_{{t+2}}) = LMHead(h_mtp)",
                "The shared output head predicts one further token. The module can be dropped at inference, or kept for speculative decoding.",
            ),
        ],
        "csa2": [
            Op(
                "query",
                f"c_Q = RMSNorm(h W_Qa) ;  q = c_Q W_Qb    [{H}] → [d_q] → [{N} × D]",
                "Queries go through a low-rank down projection and RMSNorm, then up to one vector per head. Every layer computes its own query.",
            ),
            Op(
                "window",
                "kv = RMSNorm(h W_kv) ;  RoPE on the tail ;  attend the last window",
                "A single key/value vector covers the local sliding window. This cache is per layer and is not compressed.",
            ),
            Op(
                "compress",
                "C = pool_r(kv) with a learned softmax gate",
                "Every r tokens become one global KV latent. Ratio 0 skips this. Ratio 1 keeps one latent per token. Only KV-source layers build the latent; later layers reuse it.",
            ),
            Op(
                "index",
                "topk = Indexer(q_index, K_index) over compressed positions",
                "A small indexer scores compressed positions and keeps the top ones. Full mode builds the keys. Reindex mode rescores a reused key cache. Reuse mode keeps the previous top-k.",
            ),
            Op(
                "attend",
                "o = sparse_attn(q, window KV ∪ selected compressed KV)",
                "The query reads the local window and the selected compressed entries, then a grouped low-rank projection returns the hidden size.",
            ),
        ],
        "mhc": [
            Op(
                "mix",
                "X' = B X + C F(A_prev X) ;  (A, B, C) = H(X)",
                "The residual is hc_mult parallel streams. This block applies the mix coefficients from the previous site, and predicts the coefficients the next site will use. B is Sinkhorn-normalized.",
            ),
        ],
        "engram": [
            Op(
                "hash",
                "ids = hash(n-grams of the token) across orders and heads",
                "Each head hashes recent n-grams into a huge embedding table. The lookup is a memory read, not a layer of attention.",
            ),
            Op(
                "gate",
                "h ← h + sigmoid(signed_sqrt(match(h, key))) · value",
                "The retrieved value is added to each residual stream, gated by how well the stream matches the retrieved key.",
            ),
        ],
        "dspark": [
            Op(
                "draft",
                "logits for the next block of positions, in one forward pass",
                "A few extra sliding-window layers draft several future tokens at once. A small Markov head ties those draft tokens together.",
            ),
            Op(
                "verify",
                "accept a prefix using a confidence schedule",
                "A confidence head estimates which draft prefix will survive verification. The backbone stays the source of truth.",
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
        "mla": "mla",
        "deepseek_moe": "deepseek_moe",
        "mtp": "mtp",
        "csa2": "csa2",
        "mhc": "mhc",
        "engram": "engram",
        "dspark": "dspark",
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
        "mtp": (
            [TensorSpec("h", [B, S, H]), TensorSpec("next_token", [B, S])],
            [TensorSpec("logits_t_plus_2", [B, S, V])],
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
        "mla",
        "deepseek_moe",
        "mtp",
        "csa2",
        "mhc",
        "engram",
        "dspark",
    }
)
