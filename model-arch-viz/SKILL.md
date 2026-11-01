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

1. **HF hub or local config** (default). This is the path for DeepSeek too:
   V2/V3 latent attention and V4/V4.1 compressed attention are read from the
   config fields (`text_config` included). Do not write a `.arch.json` for a
   model whose config already carries the layer sizes and the schedule.

```bash
python -m model_arch_viz from-config path/to/config.json -o architecture.html
python -m model_arch_viz from-hf <org/model> -o architecture.html
python -m model_arch_viz from-hf <org/model> --meta -o architecture.html
```

```python
from model_arch_viz import from_config_json, from_hf, show
arch = from_config_json("examples/config.json")
show(arch)  # Jupyter
```

2. **Paper**, only when the config does not contain the math. Extract an English
   IR JSON (see `examples/deepseek_v3.arch.json`), then:

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

- The full architecture stays visible on the left. Clicking a block jumps to
  its first explanation.
- The right pane walks one operation at a time: math, a plain explanation
  (`Op.note`), and that layer's interactive strip. Previous / Next (and the
  arrow keys) move through every operation in order.
- No bold / strong emphasis in the widget.
- Each layer gets an interactive view. Attention, compressed attention, RoPE,
  the embedding table, the LM head, and hyper-connection mixes are matrices.
  Widths, norms, residuals, routing, and token shifts stay one-dimensional.
- First paint already shows a real query/token position.

## Rules

- English only in IR, UI, commits, and chat summaries of papers.
- Standalone HTML and Jupyter use the same widget (`render_html` /
  `show`).
- Do not scrape PDFs inside the package; the agent writes IR from the paper.
- If only Chinese docs exist for a model, summarize into English IR.

## Kind catalog

`embedding`, `rope`, `rmsnorm`, `layernorm`, `attention`, `gqa_attention`,
`mla`, `csa2`, `mhc`, `mlp`, `swiglu_mlp`, `moe`, `deepseek_moe`,
`engram`, `residual`, `transformer_block`, `conv2d`, `vit_patch`,
`cross_attention`, `lm_head`, `mtp`, `dspark`, `dropout`, `linear`.
