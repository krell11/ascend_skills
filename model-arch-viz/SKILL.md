---
name: model-arch-viz
description: >-
  Visualize neural network architecture and per-layer math as a Jupyter or
  standalone HTML widget from Hugging Face config.json, optional safetensors
  headers, or a paper-derived IR — without downloading weights. Use when the
  user asks to visualize a model, show layer operations, explain architecture
  from a paper or HF config, or embed an architecture widget in a notebook.
---

# Model architecture visualizer

Package: `model-arch-viz` at this skill folder (editable install).

```bash
pip install -e <path-to>/model-arch-viz
```

Never call `from_pretrained` for weights. Never download full shards into memory
as tensors. Prefer `config.json`; use `--meta` only for header shapes.

## Workflow

1. **HF hub or local config** (default):

```bash
python -m model_arch_viz from-hf <org/model> -o architecture.html
python -m model_arch_viz from-hf <org/model> --meta -o architecture.html
```

```python
from model_arch_viz import from_hf, show
arch = from_hf("Qwen/Qwen2.5-0.5B", meta=False)
show(arch)  # Jupyter
```

2. **Paper**: extract an English IR JSON (see `examples/qwen2_tiny.arch.json`),
   then:

```bash
python -m model_arch_viz from-ir path/to/model.arch.json -o architecture.html
```

IR fields: `name`, `source`, `dims`, `blocks[]` with `id`, `name`, `kind`,
optional `repeat`, `inputs`/`outputs`, `ops[{name,math,note}]`, `children`.
Empty `ops` are filled from `model_arch_viz/formulas.py` by `kind`.

3. **Extend formulas** when a new `kind` appears — add a template in
   `formulas.py` (math + short explanation note + viz kind) rather than
   inventing ad-hoc HTML.

## Widget contract

- Every op shows **math** and a plain **explanation** beside it (use `Op.note`).
- No bold / strong emphasis in the widget.
- Each layer gets an interactive strip (slider recomputes cells from the
  formula): attention causal mask, embedding lookup, MLP width, MoE routing,
  norm features, residual index, RoPE neighborhood, LM head scores.
- First paint already shows a real query/token position.

## Rules

- English only in IR, UI, commits, and chat summaries of papers.
- Standalone HTML and Jupyter use the same widget (`render_html` /
  `show`).
- Do not scrape PDFs inside the package; the agent writes IR from the paper.
- If only Chinese docs exist for a model, summarize into English IR.

## Kind catalog

`embedding`, `rope`, `rmsnorm`, `layernorm`, `attention`, `gqa_attention`,
`mlp`, `swiglu_mlp`, `moe`, `residual`, `transformer_block`, `conv2d`,
`vit_patch`, `cross_attention`, `lm_head`, `dropout`, `linear`.
