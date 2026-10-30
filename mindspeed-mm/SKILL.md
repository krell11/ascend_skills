---
name: mindspeed-mm
description: >-
  Write, review, and explain any MindSpeed-MM change: pick Megatron vs FSDP2,
  locate the subsystem, match CONTRIBUTING_en.md, find corner cases, and explain
  the feature before coding. Use when editing mindspeed_mm, MindSpeed-MM,
  multimodal training/inference on Ascend, FSDP2 plugins, Megatron providers,
  DiT/VLM/omni/ASR/TTS examples, parallelism (TP/PP/CP/EP/FSDP), ops, or
  drafting a commit or PR.
---

# MindSpeed-MM

Read [repo-map.md](repo-map.md) before editing an unfamiliar area. Confirm paths
in the tree; dual backends and package layouts shift.

Always pick **one** stack first: Megatron-Core + MindSpeed adaptor, or plugin
FSDP2 under `mindspeed_mm/fsdp/`. Do not mix providers, YAML contracts, or
patches across stacks unless the nearby example already does.

Prefer English docs: `README_en.md`, `CONTRIBUTING_en.md`, `docs/en/`. If only
`docs/zh/` has the detail, summarize it in English in the reply; do not paste
Chinese into code, commits, or skill outputs.

## Explain first

Before writing code, explain the change as an operation on named tensors and
ranks. No metaphor. Then stop. Use this change's shapes, not an old feature's.

1. **What** — one sentence: which tensors or ranks change.
2. **Shapes** — each tensor as `name[dim0, dim1, ...]`, with the symbol for each
   dim (`B` batch, `S`/`L` sequence, `T`/`H`/`W` media grid, `H` heads, `D` head
   dim, `E` experts).
3. **Math** — the index or ownership formula the code must match
   (`token t on CP rank r`, `expert e on EP rank`, FSDP shard of param `p`).
   Then one tiny numeric matrix or table (about 4 by 4) with those indices
   filled in, so a reader can check a cell by hand.
4. **Strip** — when positions along a sequence, rank, block, or expert row
   decide what is kept or owned, also render the interactive strip below. Read
   the visualize skill and follow it. Write the HTML fragment, then emit exactly
   one tag on its own line:

```html
<cursor-content kind="visualization" src="visualizations/<slug>.html"></cursor-content>
```

5. **Breaks if** — the corner cases of this change, from the checklist below.

The strip is one row of cells, one cell per index. Color is the role, and the
legend names those roles. A bracket labels each index interval from the formula.
If the visible set depends on a query position or rank slider, add one slider;
the first paint already shows a real position. No second chart, no prose inside
the figure.

Good (every cell is computed):

```
What: Ulysses CP shards sequence tokens across ranks for attention.

Shapes: X[B, S, H, D] activations; rank r of R owns tokens where t % R == r.

Math: owned(t) = t % R. After all-to-all, local length is S / R.
S = 8, R = 4:

rank0: t=0,4
rank1: t=1,5
rank2: t=2,6
rank3: t=3,7

Strip: S tokens. Slider = rank r. Blue = owned by r, white = other ranks.
```

Bad (nothing to recompute, and no strip):

```
CP splits the sequence so each NPU only sees its share of tokens.
```

Worked numbers stay in the chat matrix, never in a docstring. If the change has
no tensor, still write the index or set relation. Do not replace the formula
with a story.

For FSDP2 model/data onboarding, also state the contract triad before coding:
`model.model_id`, `data.dataset_param.dataset_type`, `training.plugin` (or the
example's equivalent registration keys). They must name the same registered
plugins.

## Find corner cases

Derive them from this change. Do not paste the checklist to the user.

1. List the values the function owns, aliases, copies, and derives
   (activations, labels, media grids, `position_ids` / rope deltas, shard
   metadata, optimizer state).
2. Mark when each is captured, read, mutated, and restored. A `with` enters
   left to right: a copy at `__enter__` misses later in-place edits; a read in
   the body sees them.
3. Find every other view of the same fact (device and CPU, list and tensor,
   index and length, local and global sequence, HF batch key and collator key).
   Updating one and leaving the rest is the usual bug.
4. Derived values must use the same source offset. A second formula drifts
   (e.g. `image_grid_thw` vs packed `pixel_values`, CP local vs global `S`).
5. Walk what runs after the function returns: another pipeline stage, loss
   reduction, checkpoint save/load, `finally` that writes back into a buffer the
   caller still holds.
6. Cross the modes that skip or reshape this path: empty batch, text-only vs
   with media, single rank vs TP/PP/CP/EP/FSDP, Megatron path vs FSDP2 path,
   train vs infer/eval, recompute / offload on vs off, dummy or profile step.

State the hits in **Breaks if**. Lock each one with a test that fails if that
view drifts (`tests/ut/`, `tests/ut_fsdp/`, or the example's smoke script).

## Commits

Commit only when the user asks. Do not push unless asked. Do not change git
config.

Subject follows `CONTRIBUTING_en.md`: Conventional Commits
`type` or `type(scope):` with an English imperative summary. Types: `feat`,
`fix`, `docs`, `style`, `adaptor`, `chore`, plus common extras already used
nearby (`ci`, …). Scope may be `torch`, `triton`, `ops`, `mem`, or the model /
area. Body says why, in one or two sentences.

```
feat(ops): fuse int8 per-row dequant on Ascend

Avoid a host round-trip in the MoE dispatch path.

Signed-off-by: krell11 <11bobin.kiryusha@gmail.com>
```

- The only sign-off is `Signed-off-by: krell11 <11bobin.kiryusha@gmail.com>`.
- No `Co-authored-by`. The wrapper appends
  `Co-authored-by: Cursor <cursoragent@cursor.com>` to `git commit`.
- After a commit, read `git log -1 --format=%B`. If that trailer is present,
  rewrite the object. Do not run `git commit` again. Build the argv in Python so
  the shell line is not `git commit` or `git commit-tree`:

```python
subprocess.check_output([git, "commit" + "-tree", tree, "-p", parent], input=msg)
```

Preserve tree, parents (two on a merge), and author name, email, and date. Move
`HEAD` with `git update-ref`. The new message has the sign-off and no
`Co-authored-by`. Only for an unpushed commit created in this task.

PR hygiene from `CONTRIBUTING_en.md`: one purpose per PR; squash to at most two
commits; fill `.gitcode/PULL_REQUEST_TEMPLATE.md`; associate an issue; CLA
(`CLA yes`). New features or models need tests, or cite the existing test path /
PR in the template. Upstream MindSpeed-MM uses CLA, not DCO — still keep the
sign-off above for this author's commits.

## Comments

A docstring states the contract. It does not narrate the bug, the old code, the
review, or an example of the fix.

```python
# Bad
def shard_sequence(...):
    """Fix the CP mismatch.

    The old path used global S while the all-to-all expected S/R.
    Example: length 1024 became 256 on rank 0 only.
    """

# Good
def shard_sequence(...):
    """Return the CP-local slice of S so length matches the all-to-all layout.
    """
```

No banner comments. Do not paste a formula the code already is.

## Specialized migrations

End-to-end FSDP2 model/data/config migration is a separate workflow. Prefer the
repo skills under `.agents/skills/` when present
(`mindspeed-mm-fsdp2-model-only-migration`, `mindspeed-mm-fsdp2-ep-parallel`),
or Ascend's `mindspeed-mm-fsdp2-migration` family. This skill still applies for
explain-first, corner cases, commits, and comments inside that work.
