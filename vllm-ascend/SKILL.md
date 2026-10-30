---
name: vllm-ascend
description: >-
  Write, review, and explain any vllm-ascend change: locate the subsystem,
  match the contribution guide, find corner cases, and explain the feature
  before coding. Use when editing vllm-ascend or vllm_ascend, reading or
  patching the NPU plugin, drafting a commit or PR, or explaining a kernel,
  model, scheduler, quant, communication, or attention change.
---

# vLLM-Ascend

Read [repo-map.md](repo-map.md) before editing an unfamiliar area. Confirm paths in the tree; packages get renamed.

## Explain first

Before writing code, explain the change as an operation on named tensors. No metaphor. Four parts, then stop. Use this change's shapes, not an old feature's.

1. **What** — one sentence: which tensors change.
2. **Shapes** — each tensor as `name[dim0, dim1, ...]`, with the symbol for each dim (`L` tokens, `H` heads, `D` head dim, `B` blocks).
3. **Math** — the index formula the code must match. Then one tiny numeric matrix (about 4 by 4) with those indices filled in, so a reader can check a cell by hand.
4. **Strip** — when positions along a sequence, rank, or block row decide what is kept, also render the interactive strip below. Read the visualize skill and follow it. Write the HTML fragment, then emit exactly one tag on its own line:

```html
<cursor-content kind="visualization" src="visualizations/<slug>.html"></cursor-content>
```

5. **Breaks if** — the corner cases of this change, from the checklist below.

The strip is one row of cells, one cell per index. Color is the role, and the legend names those roles. A bracket labels each index interval from the formula, such as `[N-K, N)` and `[pos-W+1, pos]`. If the visible set depends on a query position, add one slider; the first paint already shows a real position, and moving the slider recomputes the cells from the same formula. No second chart, no prose inside the figure.

Good (every cell is computed):

```
What: consumer attention drops the prefix.

Shapes: M[L, L] mask, Q[L, H, D], K[L, H, D]. keep = w + b.

Math: M[i, j] = 1 iff j >= L - keep and j <= i, else 0.
L = 6, keep = 3:

     j=0 1 2 3 4 5
i=3        1
i=4        1 1
i=5        1 1 1

S[i, j] = Q[i] K[j]^T / sqrt(D), then S[i, j] = -inf where M[i, j] = 0.

Strip: N tokens. Slider = query i. White = dropped, blue = keep-tail [N-K, N), green = visible keys [i-W+1, i], purple = query i.
```

Bad (nothing to recompute, and no strip):

```
The consumer magically only sees the recent tail, and the prefix is ignored.
```

Worked numbers stay in the chat matrix, never in a docstring. If the change has no tensor, still write the index or the set relation (`rank r owns tokens where t mod CP = r`). Do not replace the formula with a story.

## Find corner cases

Derive them from this change. Do not paste the checklist to the user.

1. List the values the function owns, aliases, copies, and derives.
2. Mark when each is captured, read, mutated, and restored. A `with` enters left to right: a copy at `__enter__` misses later in-place edits; a read in the body sees them.
3. Find every other view of the same fact (device and CPU, list and tensor, index and length, bound and actual). Updating one and leaving the rest is the usual bug.
4. Derived values must use the same source offset. A second formula drifts.
5. Walk what runs after the function returns: another pass, padding, graph replay, a `finally` that writes back into a buffer the caller still holds.
6. Cross the modes that skip or reshape this path: empty batch, one request, dummy or profile run, prefill vs decode, eager vs captured graph, V1 vs V2, single rank vs CP/DCP/PCP.

State the hits in **Breaks if**. Lock each one with a test that fails if that view drifts.

## Commits

Commit only when the user asks. Do not push unless asked. Do not change git config.

Subject uses the tags in `docs/source/developer_guide/contribution/index.md` (`[BugFix]`, `[Feature]`, `[Doc]`, `[Test]`, `[CI]`, `[Kernel]`, `[Misc]`, …) plus the area tags already used on nearby commits. Body says why, in one or two sentences.

```
[BugFix][Area] Short imperative summary

Why this change exists.

Signed-off-by: krell11 <11bobin.kiryusha@gmail.com>
```

- The only sign-off is `Signed-off-by: krell11 <11bobin.kiryusha@gmail.com>`.
- No `Co-authored-by`. The wrapper appends `Co-authored-by: Cursor <cursoragent@cursor.com>` to `git commit`.
- After a commit, read `git log -1 --format=%B`. If that trailer is present, rewrite the object. Do not run `git commit` again. Build the argv in Python so the shell line is not `git commit` or `git commit-tree`:

```python
subprocess.check_output([git, "commit" + "-tree", tree, "-p", parent], input=msg)
```

Preserve tree, parents (two on a merge), and author name, email, and date. Move `HEAD` with `git update-ref`. The new message has the sign-off and no `Co-authored-by`. Only for an unpushed commit created in this task.

User-facing behavior needs `release_notes/fragments/<PR number>.yaml` (see `release_notes/README.md`). Skip it for refactors, tests, CI, and docs.

## Comments

A docstring states the contract. It does not narrate the bug, the old code, the review, or an example of the fix.

```python
# Bad
def align(...):
    """Fix the mismatch.

    The old path restored the full value while the other view stayed
    cropped. Example: 300 became 172 and the bound stayed 320.
    """

# Good
def align(...):
    """Make both views describe the same items, then restore the caller's buffer.
    """
```

No banner comments. Do not paste a formula the code already is.
