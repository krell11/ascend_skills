"""Self-contained HTML architecture widget.

The left column keeps the whole network visible. The right column walks one
operation at a time: math, a plain explanation, and an interactive strip.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from ..formulas import viz_kind_for
from ..ir import Architecture, Layer


def write_html(arch: Architecture, path: str | Path, *, standalone: bool = True) -> Path:
    path = Path(path)
    path.write_text(render_html(arch, standalone=standalone), encoding="utf-8")
    return path


def render_html(arch: Architecture, *, standalone: bool = True) -> str:
    nodes = _flatten(arch)
    body = _shell(arch, nodes, fill=standalone)
    if standalone:
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
            f"<title>{html.escape(arch.name)}</title>\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
            "</head>\n<body>\n"
            f"{body}\n</body>\n</html>\n"
        )
    return body


def _flatten(arch: Architecture) -> list[dict]:
    rows: list[dict] = []

    def walk(layer: Layer, depth: int) -> None:
        rows.append(
            {
                "id": layer.id,
                "label": layer.name,
                "kind": layer.kind,
                "viz": viz_kind_for(layer.kind),
                "depth": depth,
                "repeat": layer.repeat,
                "inputs": [{"name": t.name, "shape": t.shape} for t in layer.inputs],
                "outputs": [{"name": t.name, "shape": t.shape} for t in layer.outputs],
                "ops": [{"name": o.name, "math": o.math, "note": o.note} for o in layer.ops],
                "meta": layer.meta,
                "has_children": bool(layer.children),
            }
        )
        for child in layer.children:
            walk(child, depth + 1)

    for block in arch.blocks:
        walk(block, 0)
    return rows


def _shell(arch: Architecture, nodes: list[dict], *, fill: bool) -> str:
    uid = "mav-" + "".join(c if c.isalnum() else "-" for c in arch.name.lower())[:40]
    uid = uid.strip("-") or "mav-arch"
    payload = {
        "name": arch.name,
        "source": arch.source,
        "dims": arch.dims,
        "notes": arch.notes,
        "nodes": nodes,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    dims_html = "".join(
        f'<span class="mav-chip"><span class="mav-sym">{html.escape(k)}</span>'
        f" {html.escape(v)}</span>"
        for k, v in arch.dims.items()
    )
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in arch.notes)
    notes_block = f"<ul class='mav-notes'>{notes_html}</ul>" if notes_html else ""
    widget = _WIDGET
    widget = widget.replace("%%MAV_UID%%", uid)
    widget = widget.replace("%%MAV_FILL%%", " mav-fill" if fill else "")
    widget = widget.replace("%%MAV_TITLE%%", html.escape(arch.name))
    widget = widget.replace("%%MAV_SOURCE%%", html.escape(arch.source or "architecture"))
    widget = widget.replace("%%MAV_CHIPS%%", dims_html)
    widget = widget.replace("%%MAV_NOTES%%", notes_block)
    widget = widget.replace("%%MAV_DATA%%", data_json)
    return widget


_WIDGET = r"""
<div id="%%MAV_UID%%" class="mav-root%%MAV_FILL%%" data-mav="1" tabindex="0">
<style>
#%%MAV_UID%%.mav-root {
  --mav-bg: #f3f0ea;
  --mav-arch: #f6f3ee;
  --mav-panel: #fffcf8;
  --mav-ink: #1c1917;
  --mav-muted: #78716c;
  --mav-line: #e6e1d8;
  --mav-accent: #0f766e;
  --mav-accent-soft: #ccfbf1;
  --mav-select: #134e4a;
  --mav-drop: #f5f5f4;
  --mav-blue: #0284c7;
  --mav-blue-soft: #e0f2fe;
  --mav-green: #15803d;
  --mav-green-soft: #dcfce7;
  --mav-purple: #7c3aed;
  --mav-purple-soft: #ede9fe;
  --mav-amber: #b45309;
  --mav-amber-soft: #fef3c7;
  --kind: #57534e;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--mav-ink);
  background: var(--mav-bg);
  border: 1px solid var(--mav-line);
  border-radius: 16px;
  overflow: hidden;
  max-width: 1200px;
  height: 920px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  font-weight: 400;
}
#%%MAV_UID%%.mav-fill {
  max-width: none;
  height: 100vh;
  margin: 0;
  border: 0;
  border-radius: 0;
}
#%%MAV_UID%% * { box-sizing: border-box; font-weight: inherit; }
#%%MAV_UID%% b, #%%MAV_UID%% strong { font-weight: 400; }
#%%MAV_UID%% .viz-mla { --kind: #6d28d9; }
#%%MAV_UID%% .viz-attention, #%%MAV_UID%% .viz-csa2, #%%MAV_UID%% .viz-rope { --kind: #7c3aed; }
#%%MAV_UID%% .viz-mhc { --kind: #57534e; }
#%%MAV_UID%% .viz-engram { --kind: #b45309; }
#%%MAV_UID%% .viz-dspark { --kind: #9f1239; }
#%%MAV_UID%% .viz-embedding, #%%MAV_UID%% .viz-lm_head { --kind: #c2410c; }
#%%MAV_UID%% .viz-norm { --kind: #0369a1; }
#%%MAV_UID%% .viz-mlp { --kind: #15803d; }
#%%MAV_UID%% .viz-moe, #%%MAV_UID%% .viz-deepseek_moe { --kind: #0f766e; }
#%%MAV_UID%% .viz-residual { --kind: #78716c; }
#%%MAV_UID%% .viz-block { --kind: #134e4a; }
#%%MAV_UID%% .viz-mtp { --kind: #9f1239; }
#%%MAV_UID%% .viz-generic, #%%MAV_UID%% .viz-linear, #%%MAV_UID%% .viz-dropout,
#%%MAV_UID%% .viz-conv, #%%MAV_UID%% .viz-vit { --kind: #57534e; }
#%%MAV_UID%% .mav-header {
  flex: none;
  padding: 14px 18px 12px;
  border-bottom: 1px solid var(--mav-line);
  background: var(--mav-panel);
}
#%%MAV_UID%% .mav-title { margin: 0; font-size: 1.2rem; letter-spacing: -0.02em; }
#%%MAV_UID%% .mav-sub { margin: 3px 0 0; color: var(--mav-muted); font-size: 0.8rem; }
#%%MAV_UID%% .mav-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
#%%MAV_UID%% .mav-chip {
  font-size: 0.72rem;
  color: var(--mav-muted);
  background: var(--mav-bg);
  border: 1px solid var(--mav-line);
  border-radius: 999px;
  padding: 3px 9px;
}
#%%MAV_UID%% .mav-sym { color: var(--mav-accent); margin-right: 2px; }
#%%MAV_UID%% .mav-body {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
}
#%%MAV_UID%% .mav-arch {
  min-height: 0;
  overflow: auto;
  background-color: var(--mav-arch);
  background-image: radial-gradient(#e4dfd6 1px, transparent 1px);
  background-size: 16px 16px;
  border-right: 1px solid var(--mav-line);
  padding: 0 14px 16px;
}
#%%MAV_UID%% .mav-colhead {
  position: sticky;
  top: 0;
  z-index: 2;
  margin: 0 -14px 8px;
  padding: 10px 14px 8px;
  background: rgba(246, 243, 238, 0.94);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-hint {
  margin: -2px 0 10px;
  font-size: 0.75rem;
  color: var(--mav-muted);
  letter-spacing: 0;
  text-transform: none;
}
#%%MAV_UID%% .mav-link {
  height: 10px;
  margin-left: 16px;
  width: 0;
  border-left: 1px solid #c4bfb6;
  position: relative;
}
#%%MAV_UID%% .mav-link::after {
  content: "";
  position: absolute;
  left: -4px;
  bottom: -1px;
  border-width: 5px 4px 0 4px;
  border-style: solid;
  border-color: #c4bfb6 transparent transparent transparent;
}
#%%MAV_UID%% .mav-module { margin: 0 14px 2px 0; }
#%%MAV_UID%% .mav-module-body {
  background: rgba(255, 252, 248, 0.92);
  border: 1px solid var(--mav-line);
  border-radius: 14px;
  padding: 6px;
}
#%%MAV_UID%% .mav-module.is-deck .mav-module-body {
  box-shadow:
    5px 5px 0 #fff,
    6px 6px 0 #e7e5e4,
    10px 10px 0 #fff,
    11px 11px 0 #d6d3d1;
}
#%%MAV_UID%% .mav-module.is-current .mav-module-body { border-color: var(--mav-accent); }
#%%MAV_UID%% .mav-nest { display: flex; flex-direction: column; padding-left: 6px; }
#%%MAV_UID%% .mav-node {
  display: block;
  width: 100%;
  text-align: left;
  font: inherit;
  color: inherit;
  background: #fff;
  border: 1px solid var(--mav-line);
  border-left: 3px solid var(--kind);
  border-radius: 10px;
  padding: 5px 9px 5px 10px;
  cursor: pointer;
  scroll-margin-top: 48px;
}
#%%MAV_UID%% .mav-node:hover { background: #fafaf9; }
#%%MAV_UID%% .mav-node.is-on {
  background: color-mix(in srgb, var(--kind) 12%, white);
  box-shadow: 0 0 0 1px var(--kind);
}
#%%MAV_UID%% .mav-node:focus-visible, #%%MAV_UID%% .mav-navbtn:focus-visible,
#%%MAV_UID%% .mav-op-chip:focus-visible, #%%MAV_UID%% .mav-seg:focus-visible {
  outline: 2px solid var(--mav-accent);
  outline-offset: 2px;
}
#%%MAV_UID%% .mav-kicker {
  display: block;
  font-size: 0.64rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--kind);
}
#%%MAV_UID%% .mav-name { display: block; font-size: 0.86rem; margin-top: 1px; line-height: 1.25; }
#%%MAV_UID%% .mav-repeat {
  display: inline-block;
  margin-left: 6px;
  padding: 0 7px;
  border-radius: 999px;
  background: var(--mav-accent-soft);
  color: var(--mav-select);
  font-size: 0.72rem;
}
#%%MAV_UID%% .mav-io {
  display: block;
  margin-top: 3px;
  font-family: Consolas, "Cascadia Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  color: var(--mav-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#%%MAV_UID%% .mav-legend { display: flex; flex-wrap: wrap; gap: 8px 12px; padding: 14px 2px 2px; }
#%%MAV_UID%% .mav-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-legend-item i {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: var(--kind);
  display: inline-block;
}
#%%MAV_UID%% .mav-notes {
  margin: 14px 0 0;
  padding-left: 18px;
  color: var(--mav-muted);
  font-size: 0.75rem;
}
#%%MAV_UID%% .mav-notes li { margin-bottom: 4px; }
#%%MAV_UID%% .mav-explain {
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: var(--mav-panel);
}
#%%MAV_UID%% .mav-stepbar {
  flex: none;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--mav-line);
  background: var(--mav-panel);
}
#%%MAV_UID%% .mav-stepmeta { flex: 1; min-width: 0; }
#%%MAV_UID%% .mav-stepcount { font-size: 0.82rem; color: var(--mav-muted); margin-bottom: 6px; }
#%%MAV_UID%% .mav-step-now { color: var(--mav-ink); }
#%%MAV_UID%% .mav-keys { float: right; font-size: 0.72rem; }
#%%MAV_UID%% .mav-progress { display: flex; align-items: flex-end; gap: 3px; height: 14px; }
#%%MAV_UID%% .mav-seg {
  flex: 1 1 0;
  height: 7px;
  min-width: 3px;
  border: 0;
  border-radius: 99px;
  padding: 0;
  cursor: pointer;
  background: var(--kind);
  opacity: 0.35;
}
#%%MAV_UID%% .mav-seg.is-on { height: 12px; opacity: 1; }
#%%MAV_UID%% .mav-navbtn {
  font: inherit;
  font-size: 0.85rem;
  border: 1px solid var(--mav-line);
  background: #fff;
  color: var(--mav-ink);
  border-radius: 999px;
  padding: 8px 14px;
  cursor: pointer;
}
#%%MAV_UID%% .mav-navbtn:hover:not(:disabled) { border-color: var(--mav-accent); color: var(--mav-select); }
#%%MAV_UID%% .mav-navbtn:disabled { opacity: 0.35; cursor: default; }
#%%MAV_UID%% .mav-explain-body { flex: 1; min-height: 0; overflow: auto; padding: 16px 18px 28px; }
#%%MAV_UID%% .mav-explain-body h2 { margin: 0 0 4px; font-size: 1.15rem; letter-spacing: -0.02em; }
#%%MAV_UID%% .mav-kind { color: var(--mav-muted); font-size: 0.8rem; }
#%%MAV_UID%% .mav-op-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
#%%MAV_UID%% .mav-op-chip {
  font: inherit;
  font-size: 0.75rem;
  border: 1px solid var(--mav-line);
  background: #fff;
  color: var(--mav-muted);
  border-radius: 999px;
  padding: 4px 10px;
  cursor: pointer;
}
#%%MAV_UID%% .mav-op-chip.is-on {
  background: var(--mav-accent-soft);
  border-color: var(--mav-accent);
  color: var(--mav-select);
}
#%%MAV_UID%% .mav-shapes { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
#%%MAV_UID%% .mav-shape {
  background: var(--mav-bg);
  border: 1px solid var(--mav-line);
  border-radius: 10px;
  padding: 6px 10px;
  font-size: 0.8rem;
}
#%%MAV_UID%% .mav-shape span { color: var(--mav-muted); margin-right: 6px; }
#%%MAV_UID%% .mav-mathblock {
  margin-top: 14px;
  background: #fff;
  border: 1px solid var(--mav-line);
  border-radius: 12px;
  padding: 14px 16px;
}
#%%MAV_UID%% .mav-op-name {
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--mav-accent);
  margin-bottom: 6px;
}
#%%MAV_UID%% .mav-op-math {
  font-family: "Cambria Math", "Times New Roman", Times, serif;
  font-size: 1.18rem;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
#%%MAV_UID%% .mav-prose {
  margin: 12px 0 0;
  font-size: 0.98rem;
  line-height: 1.55;
  color: var(--mav-ink);
}
#%%MAV_UID%% .mav-play {
  margin-top: 16px;
  background: var(--mav-bg);
  border: 1px solid var(--mav-line);
  border-radius: 12px;
  padding: 12px 14px 14px;
}
#%%MAV_UID%% .mav-play-title {
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--mav-muted);
  margin-bottom: 8px;
}
#%%MAV_UID%% .mav-play-formula {
  font-family: "Cambria Math", "Times New Roman", Times, serif;
  font-size: 0.95rem;
  line-height: 1.4;
  margin-bottom: 10px;
}
#%%MAV_UID%% .mav-play-note { margin: 8px 0 0; font-size: 0.75rem; color: var(--mav-muted); }
#%%MAV_UID%% .mav-controls { display: flex; align-items: center; margin: 2px 0 12px; }
#%%MAV_UID%% .mav-controls label {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  font-size: 0.85rem;
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-controls input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  width: auto;
  min-width: 0;
  height: 36px;
  margin: 0;
  background: transparent;
  cursor: grab;
  touch-action: none;
}
#%%MAV_UID%% .mav-controls input[type="range"]:active { cursor: grabbing; }
#%%MAV_UID%% .mav-controls input[type="range"]::-webkit-slider-runnable-track {
  height: 8px;
  background: #e7e5e4;
  border-radius: 99px;
}
#%%MAV_UID%% .mav-controls input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 26px;
  height: 26px;
  margin-top: -9px;
  border-radius: 50%;
  background: var(--mav-accent);
  border: 3px solid #fff;
  box-shadow: 0 1px 4px rgba(28, 25, 23, 0.35);
}
#%%MAV_UID%% .mav-controls input[type="range"]::-moz-range-track {
  height: 8px;
  background: #e7e5e4;
  border: 0;
  border-radius: 99px;
}
#%%MAV_UID%% .mav-controls input[type="range"]::-moz-range-thumb {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--mav-accent);
  border: 3px solid #fff;
  box-shadow: 0 1px 4px rgba(28, 25, 23, 0.35);
}
#%%MAV_UID%% .mav-readout {
  color: var(--mav-ink);
  min-width: 2.2ch;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
#%%MAV_UID%% .mav-section-label {
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--mav-muted);
  margin-bottom: 6px;
}
#%%MAV_UID%% .mav-strip { display: flex; flex-wrap: nowrap; gap: 3px; overflow-x: auto; padding-bottom: 4px; }
#%%MAV_UID%% .mav-cell {
  flex: 0 0 auto;
  min-width: 28px;
  height: 28px;
  padding: 0 4px;
  border-radius: 5px;
  border: 1px solid var(--mav-line);
  background: var(--mav-drop);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-cell.role-query { background: var(--mav-purple-soft); border-color: var(--mav-purple); color: var(--mav-purple); }
#%%MAV_UID%% .mav-cell.role-visible { background: var(--mav-green-soft); border-color: var(--mav-green); color: var(--mav-green); }
#%%MAV_UID%% .mav-cell.role-keep { background: var(--mav-blue-soft); border-color: var(--mav-blue); color: var(--mav-blue); }
#%%MAV_UID%% .mav-cell.role-active { background: var(--mav-accent-soft); border-color: var(--mav-accent); color: var(--mav-select); }
#%%MAV_UID%% .mav-cell.role-gate { background: var(--mav-amber-soft); border-color: var(--mav-amber); color: var(--mav-amber); }
#%%MAV_UID%% .mav-cell.role-dropped { background: var(--mav-drop); color: #a8a29e; }
#%%MAV_UID%% .mav-brackets { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; font-size: 0.75rem; color: var(--mav-muted); }
#%%MAV_UID%% .mav-bracket span {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 5px;
  vertical-align: -1px;
  border: 1px solid var(--mav-line);
}
#%%MAV_UID%% .mav-bracket .q { background: var(--mav-purple-soft); border-color: var(--mav-purple); }
#%%MAV_UID%% .mav-bracket .v { background: var(--mav-green-soft); border-color: var(--mav-green); }
#%%MAV_UID%% .mav-bracket .k { background: var(--mav-blue-soft); border-color: var(--mav-blue); }
#%%MAV_UID%% .mav-bracket .a { background: var(--mav-accent-soft); border-color: var(--mav-accent); }
#%%MAV_UID%% .mav-bracket .g { background: var(--mav-amber-soft); border-color: var(--mav-amber); }
#%%MAV_UID%% .mav-bracket .d { background: var(--mav-drop); }
#%%MAV_UID%% .mav-matrix { margin-top: 10px; overflow-x: auto; }
#%%MAV_UID%% .mav-matrix table { border-collapse: collapse; font-size: 0.72rem; }
#%%MAV_UID%% .mav-matrix th, #%%MAV_UID%% .mav-matrix td {
  width: 22px;
  height: 22px;
  text-align: center;
  border: 1px solid var(--mav-line);
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-matrix td.on, #%%MAV_UID%% .mav-matrix td.m-on { background: var(--mav-green-soft); color: var(--mav-green); }
#%%MAV_UID%% .mav-matrix td.query-row, #%%MAV_UID%% .mav-matrix tr.is-query th { color: var(--mav-purple); }
#%%MAV_UID%% .mav-matrix td.m-query { background: var(--mav-purple-soft); color: var(--mav-purple); }
#%%MAV_UID%% .mav-matrix td.m-active { background: var(--mav-accent-soft); color: var(--mav-select); }
#%%MAV_UID%% .mav-matrix td.m-keep { background: var(--mav-blue-soft); color: var(--mav-blue); }
#%%MAV_UID%% .mav-matrix td.m-off { background: var(--mav-drop); color: #a8a29e; }
#%%MAV_UID%% .mav-matrix.is-num th, #%%MAV_UID%% .mav-matrix.is-num td { width: auto; min-width: 34px; padding: 2px 4px; }
#%%MAV_UID%% .mav-matrices { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }
#%%MAV_UID%% .mav-bar-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
  margin: 5px 0;
  font-size: 0.75rem;
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-bar-label { min-width: 88px; }
#%%MAV_UID%% .mav-bar-row.is-focus .mav-bar-label { color: var(--mav-ink); }
#%%MAV_UID%% .mav-bar-track { height: 10px; background: #f5f5f4; border-radius: 99px; overflow: hidden; }
#%%MAV_UID%% .mav-bar-fill { display: block; height: 100%; border-radius: 99px; background: #a8a29e; }
#%%MAV_UID%% .mav-bar-fill.role-keep { background: #0284c7; }
#%%MAV_UID%% .mav-bar-fill.role-query { background: #7c3aed; }
#%%MAV_UID%% .mav-bar-fill.role-active { background: #0f766e; }
#%%MAV_UID%% .mav-bar-fill.role-gate { background: #b45309; }
#%%MAV_UID%% .mav-tag {
  display: inline-block;
  margin-left: 4px;
  padding: 0 5px;
  border-radius: 4px;
  background: var(--mav-accent-soft);
  color: var(--mav-select);
  font-size: 0.66rem;
}
#%%MAV_UID%% .mav-groups { display: grid; gap: 4px; }
#%%MAV_UID%% .mav-group {
  border: 1px solid var(--mav-line);
  border-radius: 8px;
  padding: 8px 2px;
  text-align: center;
  font-size: 0.72rem;
  background: #f5f5f4;
  color: var(--mav-muted);
}
#%%MAV_UID%% .mav-group.is-on {
  background: var(--mav-accent-soft);
  border-color: var(--mav-accent);
  color: var(--mav-select);
}
#%%MAV_UID%% .mav-flowlet {
  display: flex;
  gap: 8px;
  align-items: center;
  background: #fff;
  border: 1px solid var(--mav-line);
  border-radius: 8px;
  padding: 8px 10px;
  margin-top: 6px;
  font-size: 0.85rem;
}
#%%MAV_UID%% .mav-flowlet span {
  width: 18px;
  height: 18px;
  border-radius: 99px;
  background: var(--mav-accent-soft);
  color: var(--mav-select);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  flex: none;
}
#%%MAV_UID%% .mav-empty { padding: 8px 0; color: var(--mav-muted); }
@media (max-width: 800px) {
  #%%MAV_UID%%.mav-root, #%%MAV_UID%%.mav-fill { height: auto; min-height: 100vh; }
  #%%MAV_UID%% .mav-body { grid-template-columns: 1fr; }
  #%%MAV_UID%% .mav-arch { max-height: 48vh; border-right: 0; border-bottom: 1px solid var(--mav-line); }
  #%%MAV_UID%% .mav-explain { min-height: 70vh; }
  #%%MAV_UID%% .mav-keys { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  #%%MAV_UID%% .mav-cell { transition: none; }
}
</style>
  <div class="mav-header">
    <h1 class="mav-title">%%MAV_TITLE%%</h1>
    <p class="mav-sub">%%MAV_SOURCE%%</p>
    <div class="mav-chips">%%MAV_CHIPS%%</div>
  </div>
  <div class="mav-body">
    <div class="mav-arch">
      <div class="mav-colhead">Architecture<div class="mav-hint">The whole network. The highlight follows the explanation.</div></div>
      <div class="mav-diagram"></div>
      <div class="mav-legend"></div>
      %%MAV_NOTES%%
    </div>
    <div class="mav-explain">
      <div class="mav-stepbar">
        <button type="button" class="mav-navbtn" data-mav-prev>Previous</button>
        <div class="mav-stepmeta">
          <div class="mav-stepcount" data-mav-count></div>
          <div class="mav-progress" data-mav-progress></div>
        </div>
        <button type="button" class="mav-navbtn" data-mav-next>Next</button>
      </div>
      <div class="mav-explain-body" aria-live="polite"></div>
    </div>
  </div>
  <script type="application/json" id="%%MAV_UID%%-data">%%MAV_DATA%%</script>
  <script>
(function() {
  var root = document.getElementById("%%MAV_UID%%");
  if (!root) return;
  var data = JSON.parse(document.getElementById("%%MAV_UID%%-data").textContent);
  var arch = root.querySelector(".mav-arch");
  var diagram = root.querySelector(".mav-diagram");
  var legend = root.querySelector(".mav-legend");
  var explainBody = root.querySelector(".mav-explain-body");
  var countEl = root.querySelector("[data-mav-count]");
  var progress = root.querySelector("[data-mav-progress]");
  var prevBtn = root.querySelector("[data-mav-prev]");
  var nextBtn = root.querySelector("[data-mav-next]");
  if (!data.nodes || !data.nodes.length) {
    explainBody.innerHTML = '<p class="mav-empty">No layers in this architecture.</p>';
    return;
  }

  var SEQ = 12;
  var state = { step: 0, query: 7, token: 3 };
  var playFrame = 0;
  var steps = [];
  data.nodes.forEach(function(node, idx) {
    var ops = (node.ops && node.ops.length) ? node.ops : [{
      name: node.kind,
      math: "",
      note: "No formula is stored for this block yet."
    }];
    ops.forEach(function(op, j) {
      steps.push({ node: idx, op: j, opData: op });
    });
  });

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function(c) {
      return ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c];
    });
  }
  function safeClass(v) {
    return String(v || "generic").replace(/[^a-z0-9_]/g, "") || "generic";
  }
  function num(v, fallback) {
    var n = Number(v);
    return isFinite(n) && n > 0 ? n : fallback;
  }
  function shapeText(arr) {
    return "[" + (arr || []).join(", ") + "]";
  }
  function ioLine(node) {
    function side(list) {
      return (list || []).map(function(t) {
        return t.name + " " + shapeText(t.shape);
      }).join(", ");
    }
    var a = side(node.inputs);
    var b = side(node.outputs);
    if (a && b) return a + "  →  " + b;
    return a || b;
  }
  function cell(i, role, label) {
    return '<div class="mav-cell role-' + role + '" title="' + esc(String(label != null ? label : i)) + '">' + esc(String(i)) + '</div>';
  }
  function legendHtml(items) {
    return '<div class="mav-brackets">' + items.map(function(it) {
      return '<span class="mav-bracket"><span class="' + it.cls + '"></span>' + esc(it.text) + '</span>';
    }).join("") + '</div>';
  }
  function slider(kind, label, value, max) {
    var attr = kind === "query" ? "data-mav-query" : "data-mav-token";
    var ro = kind === "query" ? "data-mav-qval" : "data-mav-tval";
    return '<div class="mav-controls"><label>' + esc(label) +
      ' <input type="range" min="0" max="' + max + '" step="1" value="' + value + '" ' + attr + '/>' +
      '<span class="mav-readout" ' + ro + '>' + value + '</span></label></div>';
  }
  function barRow(label, value, max, role, focus, cached) {
    var pct = Math.max(1.5, Math.round(1000 * value / max) / 10);
    return '<div class="mav-bar-row' + (focus ? " is-focus" : "") + '">' +
      '<span class="mav-bar-label">' + esc(label) + (cached ? ' <span class="mav-tag">cache</span>' : "") + '</span>' +
      '<span class="mav-bar-track"><span class="mav-bar-fill role-' + role + '" style="width:' + pct + '%"></span></span>' +
      '<span class="mav-bar-val">' + esc(String(value)) + '</span></div>';
  }
  function maskMatrix(spec) {
    var rows = spec.rows, cols = spec.cols;
    var html = '<div class="mav-matrix' + (spec.numeric ? " is-num" : "") + '"><table aria-label="' + esc(spec.aria || "matrix") + '"><thead><tr><th>' + esc(spec.corner || "") + "</th>";
    for (var c = 0; c < cols; c++) html += "<th>" + esc(spec.col ? spec.col(c) : c) + "</th>";
    html += "</tr></thead><tbody>";
    for (var r = 0; r < rows; r++) {
      html += '<tr' + (spec.query === r ? ' class="is-query"' : "") + "><th>" + esc(spec.row ? spec.row(r) : r) + "</th>";
      for (var c2 = 0; c2 < cols; c2++) {
        var mark = spec.mark(r, c2);
        var text = spec.text ? spec.text(r, c2, mark) : (mark === "off" ? "·" : "1");
        html += '<td class="m-' + mark + '">' + esc(text) + "</td>";
      }
      html += "</tr>";
    }
    return html + "</tbody></table></div>";
  }
  function causalMatrix(query) {
    return maskMatrix({
      rows: SEQ, cols: SEQ, corner: "q\\k", aria: "Causal attention mask", query: query,
      mark: function(r, c) {
        if (c > r) return "off";
        if (r === query && c === r) return "query";
        if (r === query) return "on";
        return "keep";
      },
      text: function(r, c, mark) { return mark === "off" ? "·" : "1"; }
    });
  }
  function tokenStrip(token, activeRole) {
    var html = '<div class="mav-strip" aria-label="Sequence">';
    for (var t = 0; t < SEQ; t++) html += cell(t, t === token ? (activeRole || "active") : "dropped");
    return html + "</div>";
  }

  function attentionPlay(query) {
    return '<div class="mav-play-formula">M[i, j] = 1 when j ≤ i. Softmax runs over the visible keys only.</div>' +
      slider("query", "query i", query, SEQ - 1) +
      '<div class="mav-section-label">rows are queries, columns are keys</div>' +
      causalMatrix(query) +
      legendHtml([
        {cls:"q", text:"query i on the diagonal"},
        {cls:"v", text:"keys this query can see"},
        {cls:"k", text:"visible for other queries"},
        {cls:"d", text:"masked"}
      ]);
  }
  function embeddingPlay(token) {
    var dims = 8;
    var html = '<div class="mav-play-formula">The token id selects one row of the embedding matrix.</div>';
    html += slider("token", "token t", token, SEQ - 1);
    html += '<div class="mav-section-label">rows are positions, columns are hidden dims</div>';
    html += maskMatrix({
      rows: SEQ, cols: dims, corner: "t\\d", aria: "Embedding table", query: token,
      mark: function(r) { return r === token ? "active" : "keep"; },
      text: function(r, c) { return String(Math.floor(unitHash(r * 17.3 + c * 4.1) * 9)); }
    });
    html += legendHtml([
      {cls:"a", text:"row written to h[t]"},
      {cls:"k", text:"other rows of the table"}
    ]);
    return html;
  }
  function mlpPlay(token) {
    var html = '<div class="mav-play-formula">Each token expands H → I, applies the activation, then contracts back to H.</div>';
    html += slider("token", "token t", token, SEQ - 1);
    html += tokenStrip(token, "query");
    html += '<div class="mav-section-label" style="margin-top:10px">width at this token</div><div class="mav-strip">';
    for (var a = 0; a < 4; a++) html += cell("H", "keep");
    html += cell("→", "dropped");
    for (var b = 0; b < 8; b++) html += cell("I", "gate");
    html += cell("→", "dropped");
    for (var c = 0; c < 4; c++) html += cell("H", "active");
    html += "</div>";
    html += legendHtml([
      {cls:"q", text:"token t"},
      {cls:"k", text:"hidden H"},
      {cls:"g", text:"intermediate I"},
      {cls:"a", text:"output H"}
    ]);
    return html;
  }
  function moePlay(token) {
    var E = 8, topk = 2, scores = [];
    for (var e = 0; e < E; e++) scores.push({e:e, s:(Math.sin(token * 1.7 + e) + 1) / 2});
    scores.sort(function(a, b) { return b.s - a.s; });
    var chosen = {};
    for (var k = 0; k < topk; k++) chosen[scores[k].e] = true;
    var html = '<div class="mav-play-formula">Route the token to the top-' + topk + " of " + E + " experts and mix their outputs.</div>";
    html += slider("token", "token t", token, SEQ - 1);
    html += '<div class="mav-strip" aria-label="Experts">';
    for (var i = 0; i < E; i++) html += cell(i, chosen[i] ? "active" : "dropped", "expert " + i);
    html += "</div>";
    html += legendHtml([{cls:"a", text:"top-k experts"}, {cls:"d", text:"idle experts"}]);
    return html;
  }
  function unitHash(n) {
    var x = Math.sin(n) * 43758.5453;
    return x - Math.floor(x);
  }
  function route(token, E, groups, topk, topkGroup) {
    var per = Math.max(1, Math.floor(E / groups));
    var scores = [];
    for (var e = 0; e < E; e++) {
      scores.push({
        e: e,
        g: Math.min(groups - 1, Math.floor(e / per)),
        s: unitHash(token * 12.9898 + e * 78.233)
      });
    }
    var gbest = [];
    for (var g = 0; g < groups; g++) {
      var best = -1;
      for (var i = 0; i < scores.length; i++) if (scores[i].g === g && scores[i].s > best) best = scores[i].s;
      gbest.push({g: g, s: best});
    }
    gbest.sort(function(a, b) { return b.s - a.s; });
    var chosenG = {};
    for (var k = 0; k < Math.min(topkGroup, groups); k++) chosenG[gbest[k].g] = true;
    var pool = scores.filter(function(x) { return chosenG[x.g]; });
    pool.sort(function(a, b) { return b.s - a.s; });
    return {groups: chosenG, experts: pool.slice(0, topk)};
  }
  function deepseekMoePlay(node, token) {
    var meta = node.meta || {};
    var E = num(meta.n_routed_experts || meta.num_experts, 256);
    var topk = num(meta.num_experts_per_tok, 8);
    var groups = meta.n_group != null ? num(meta.n_group, 1) : 8;
    var topkGroup = meta.topk_group != null ? num(meta.topk_group, 1) : Math.min(4, groups);
    var shared = num(meta.n_shared_experts, 1);
    var scale = meta.routed_scaling_factor != null ? meta.routed_scaling_factor : 2.5;
    var scoreName = meta.scoring_func || "sigmoid";
    var sel = route(token, E, Math.max(groups, 1), topk, Math.max(topkGroup, 1));
    var html = '<div class="mav-play-formula">The shared expert always runs. ' + topk + " of " + E +
      " routed experts, scored with " + esc(scoreName) + ".";
    if (groups > 1) html += " Chosen inside " + topkGroup + " of " + groups + " groups.";
    html += " The routed mix is scaled by " + scale + ".</div>";
    html += slider("token", "token t", token, SEQ - 1);
    html += '<div class="mav-section-label">shared expert</div><div class="mav-strip">';
    for (var s = 0; s < shared; s++) html += cell(shared > 1 ? s : "S", "active", "shared expert");
    html += "</div>";
    if (groups > 1) {
      html += '<div class="mav-section-label" style="margin-top:10px">expert groups</div>';
      html += '<div class="mav-groups" style="grid-template-columns:repeat(' + groups + ',minmax(0,1fr))">';
      for (var g = 0; g < groups; g++) {
        html += '<div class="mav-group' + (sel.groups[g] ? " is-on" : "") + '">G' + g + "</div>";
      }
      html += "</div>";
    }
    html += '<div class="mav-section-label" style="margin-top:10px">routed expert ids</div><div class="mav-strip">';
    sel.experts.forEach(function(ex) { html += cell(ex.e, "gate", "expert " + ex.e); });
    html += "</div>";
    html += legendHtml([
      {cls:"a", text:"shared, always on"},
      {cls:"g", text:"routed experts for this token"}
    ]);
    html += '<p class="mav-play-note">Illustrative scores for this token, not the trained router. Moving the slider changes which groups and experts light up.</p>';
    return html;
  }
  function mlaPlay(node, query, opName) {
    var meta = node.meta || {};
    var hidden = num(meta.hidden_size, 7168);
    var qRank = num(meta.q_lora_rank, 1536);
    var kvRank = num(meta.kv_lora_rank, 512);
    var rope = num(meta.qk_rope_head_dim, 64);
    var nope = num(meta.qk_nope_head_dim, 128);
    var heads = num(meta.num_attention_heads, 128);
    var focusAttn = opName === "scores" || opName === "out" || opName === "mix";
    var html = '<div class="mav-play-formula">Inference caches c_KV (' + kvRank + ") and k_R (" + rope +
      ") per token, instead of " + heads + " full key and value vectors.</div>";
    if (!focusAttn) {
      html += barRow("h", hidden, hidden, "keep", false, false);
      html += barRow("c_Q", qRank, hidden, "query", opName === "q_down" || opName === "q_up", false);
      html += barRow("c_KV", kvRank, hidden, "active", opName === "kv_down" || opName === "kv_up", true);
      html += barRow("k_R", rope, hidden, "gate", opName === "rope_key", true);
      html += '<div class="mav-section-label" style="margin-top:10px">one head after the up projection</div><div class="mav-strip">';
      for (var i = 0; i < 8; i++) html += cell("C", "keep", "content dim");
      for (var j = 0; j < 4; j++) html += cell("R", "query", "RoPE dim");
      html += "</div>";
      html += legendHtml([
        {cls:"k", text:"content " + nope},
        {cls:"q", text:"RoPE " + rope + ", position lives here"}
      ]);
    }
    html += '<div class="mav-section-label" style="margin-top:12px">query × key mask</div>';
    html += slider("query", "query i", query, SEQ - 1);
    html += causalMatrix(query);
    html += legendHtml([
      {cls:"q", text:"query i"},
      {cls:"v", text:"keys this query attends"},
      {cls:"k", text:"visible for other queries"},
      {cls:"d", text:"future keys"}
    ]);
    return html;
  }
  function csa2Play(node, query) {
    var meta = node.meta || {};
    var ratio = Number(meta.compress_ratio);
    if (!isFinite(ratio) || ratio < 0) ratio = 0;
    var window = num(meta.sliding_window, 128);
    var topk = num(meta.index_topk, 512);
    var mode = meta.mode || (ratio === 0 ? "swa" : "full");
    var html = '<div class="mav-play-formula">Mode ' + esc(mode) + ". Local window " + window + ".";
    if (ratio > 0) html += " Global KV is compressed " + ratio + "-to-1, then the indexer keeps " + topk + " entries.";
    else html += " No compressed global KV on this layer.";
    if (meta.candidate_topk_blocks) {
      html += " Candidate pool: " + num(meta.candidate_topk_blocks, 1) + " blocks of " +
        num(meta.candidate_block_size, 1) + ".";
    }
    html += "</div>";
    html += slider("query", "query i", query, SEQ - 1);
    var draw = Math.min(3, SEQ - 1);
    html += '<div class="mav-section-label">query × key. Picture window is ' + draw + "; config window is " + window + ".</div>";
    html += maskMatrix({
      rows: SEQ, cols: SEQ, corner: "q\\k", aria: "Sliding window mask", query: query,
      mark: function(r, c) {
        var inside = c <= r && r - c <= draw;
        if (!inside) return "off";
        if (r === query && c === r) return "query";
        if (r === query) return "on";
        return "keep";
      },
      text: function(r, c, mark) { return mark === "off" ? "·" : "1"; }
    });
    html += legendHtml([
      {cls:"q", text:"query i"},
      {cls:"v", text:"inside this query's window"},
      {cls:"k", text:"inside some other window"},
      {cls:"d", text:"outside the window"}
    ]);
    if (ratio > 0) {
      var step = Math.max(ratio, 1);
      var slots = Math.max(2, Math.ceil(SEQ / step));
      function slotSeen(r, s) {
        return Math.min(SEQ - 1, (s + 1) * step - 1) <= r;
      }
      function kept(r, s) {
        if (!slotSeen(r, s)) return false;
        var visible = [];
        for (var i = 0; i < slots; i++) if (slotSeen(r, i)) visible.push(i);
        visible.sort(function(a, b) { return unitHash(r * 3.1 + b * 9.7) - unitHash(r * 3.1 + a * 9.7); });
        return visible.slice(0, Math.min(2, visible.length)).indexOf(s) !== -1;
      }
      html += '<div class="mav-section-label" style="margin-top:10px">query × compressed slot. Picture keeps 2; config index_topk is ' + topk + ".</div>";
      html += maskMatrix({
        rows: SEQ, cols: slots, corner: "q\\c", aria: "Compressed attention mask", query: query,
        mark: function(r, s) {
          if (!slotSeen(r, s)) return "off";
          if (kept(r, s)) return r === query ? "active" : "on";
          return "keep";
        },
        text: function(r, s, mark) { return mark === "active" || mark === "on" ? "1" : "·"; }
      });
      html += legendHtml([
        {cls:"a", text:"kept by the indexer for query i"},
        {cls:"v", text:"kept for another query"},
        {cls:"k", text:"compressed, not selected"},
        {cls:"d", text:"not yet visible"}
      ]);
    }
    return html;
  }
  function mhcPlay(node) {
    var meta = node.meta || {};
    var streams = num(meta.hc_mult, 4);
    var iters = num(meta.hc_sinkhorn_iters, 20);
    var html = '<div class="mav-play-formula">' + streams + "×" + streams + " mix matrices. This site applies the previous coefficients and writes the next ones. B is Sinkhorn-normalized for " + iters + " steps.</div>";
    function weight(i, j, diag) {
      if (streams === 1) return "1";
      var peak = diag;
      var off = (1 - peak) / (streams - 1);
      var raw = i === j ? peak : off;
      return raw.toFixed(2);
    }
    function oneMatrix(name, aria, diag, hot) {
      return '<div><div class="mav-section-label">' + name + '</div>' + maskMatrix({
        rows: streams, cols: streams, corner: "", aria: aria, numeric: true,
        mark: function(r, c) { return r === c ? hot : "keep"; },
        text: function(r, c) { return weight(r, c, diag); }
      }) + "</div>";
    }
    html += '<div class="mav-matrices">';
    html += oneMatrix("A, into the sublayer", "Input mix", 0.55, "query");
    html += oneMatrix("B, residual", "Residual mix", 0.70, "active");
    html += oneMatrix("C, back to the residual", "Output mix", 0.55, "on");
    html += "</div>";
    html += '<p class="mav-play-note">Illustrative coefficients. Each row sums to 1. B is doubly stochastic.</p>';
    return html;
  }
  function engramPlay(node, token) {
    var meta = node.meta || {};
    var maxN = num(meta.engram_max_ngram_size, 4);
    var heads = num(meta.engram_n_heads, 8);
    var dim = num(meta.engram_head_dim, 256);
    var html = '<div class="mav-play-formula">' + heads + " heads of width " + dim +
      ". Hash n-grams of order 2 through " + maxN + " ending at this token, then gate the retrieved value into the residual.</div>";
    html += slider("token", "token t", token, SEQ - 1);
    html += '<div class="mav-section-label">n-gram orders read at t</div><div class="mav-strip">';
    for (var n = 2; n <= maxN; n++) {
      var nrole = n === maxN ? "active" : "keep";
      html += cell(n, nrole, n + "-gram");
    }
    html += "</div>";
    html += '<div class="mav-section-label" style="margin-top:10px">tokens in the longest n-gram</div><div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) {
      var role = "dropped";
      if (t === token) role = "active";
      else if (t < token && token - t < maxN) role = "visible";
      html += cell(t, role);
    }
    html += "</div>";
    html += legendHtml([
      {cls:"a", text:"current token"},
      {cls:"v", text:"earlier tokens in the n-gram"}
    ]);
    return html;
  }
  function dsparkPlay(node, token) {
    var meta = node.meta || {};
    var block = num(meta.dspark_block_size, 5);
    var hi = Math.max(0, SEQ - block - 1);
    var t0 = Math.min(token, hi);
    var html = '<div class="mav-play-formula">One draft pass proposes the next ' + block + ' tokens. Verification keeps a prefix of that block.</div>';
    html += slider("token", "position t", t0, hi);
    html += '<div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) {
      var role = "dropped";
      if (t === t0) role = "keep";
      else if (t > t0 && t <= t0 + block) role = "gate";
      html += cell(t, role);
    }
    html += "</div>";
    html += legendHtml([
      {cls:"k", text:"last accepted token"},
      {cls:"g", text: block + " drafted positions"}
    ]);
    return html;
  }
  function mtpPlay(token) {
    var t0 = Math.min(token, SEQ - 3);
    var html = '<div class="mav-play-formula">Depth 1 reads h[t] and the embedding of token t+1, then predicts token t+2.</div>';
    html += slider("token", "position t", t0, SEQ - 3);
    html += '<div class="mav-strip" aria-label="MTP positions">';
    for (var t = 0; t < SEQ; t++) {
      var role = "dropped";
      if (t === t0) role = "keep";
      else if (t === t0 + 1) role = "gate";
      else if (t === t0 + 2) role = "active";
      html += cell(t, role);
    }
    html += "</div>";
    html += legendHtml([
      {cls:"k", text:"h[t] from the main model"},
      {cls:"g", text:"Embed(token t+1), fed in"},
      {cls:"a", text:"predicted token t+2"}
    ]);
    return html;
  }
  function normPlay(token) {
    var html = '<div class="mav-play-formula">Features of one token are rescaled together.</div>';
    html += slider("token", "token t", token, SEQ - 1);
    html += tokenStrip(token, "active");
    html += '<div class="mav-section-label" style="margin-top:10px">feature axis</div><div class="mav-strip">';
    for (var d = 0; d < 12; d++) html += cell(d, "keep");
    html += "</div>";
    html += legendHtml([{cls:"a", text:"token t"}, {cls:"k", text:"features normalized together"}]);
    return html;
  }
  function residualPlay(token) {
    var html = '<div class="mav-play-formula">Y[t] = X[t] + F(X)[t]. The same index is kept.</div>';
    html += slider("token", "token t", token, SEQ - 1);
    html += tokenStrip(token, "query");
    html += legendHtml([
      {cls:"q", text:"X[t] kept"},
      {cls:"a", text:"F(X)[t] added at the same index"}
    ]);
    return html;
  }
  function blockPlay(token) {
    var html = '<div class="mav-play-formula">Two residual sublayers. Attention mixes along the sequence. The FFN stays inside the token.</div>';
    html += slider("token", "token t", token, SEQ - 1);
    html += tokenStrip(token, "query");
    html += '<div class="mav-flowlet"><span>1</span>Norm → attention → add back</div>';
    html += '<div class="mav-flowlet"><span>2</span>Norm → FFN or MoE → add back</div>';
    return html;
  }
  function lmPlay(token) {
    var bins = 8;
    function hot(r, v) { return v === (r * 3) % bins || v === (r * 3 + 1) % bins; }
    var html = '<div class="mav-play-formula">logits[t, v] = h[t] · W_lm[:, v]</div>';
    html += slider("token", "position t", token, SEQ - 1);
    html += '<div class="mav-section-label">rows are positions, columns are vocab bins</div>';
    html += maskMatrix({
      rows: SEQ, cols: bins, corner: "t\\v", aria: "Logit scores", query: token,
      mark: function(r, v) {
        if (r === token && hot(r, v)) return "active";
        if (hot(r, v)) return "on";
        return "keep";
      },
      text: function(r, v, mark) { return mark === "keep" ? "·" : "1"; }
    });
    html += legendHtml([
      {cls:"a", text:"higher scores at position t"},
      {cls:"v", text:"higher scores at other positions"},
      {cls:"k", text:"other vocab bins"}
    ]);
    return html;
  }
  function ropePlay(token) {
    var html = '<div class="mav-play-formula">Pairs of dimensions rotate by an angle that grows with position, so |i − j| shows up in the Q·K matrix.</div>';
    html += slider("token", "position t", token, SEQ - 1);
    html += '<div class="mav-section-label">rows are queries, columns are keys, cell is |i − j|</div>';
    html += maskMatrix({
      rows: SEQ, cols: SEQ, corner: "q\\k", aria: "RoPE distance", query: token,
      mark: function(r, c) {
        var d = Math.abs(r - c);
        if (r === token && c === token) return "query";
        if (r === token && d <= 2) return "on";
        if (d === 0) return "active";
        if (d <= 2) return "keep";
        return "off";
      },
      text: function(r, c) { return String(Math.abs(r - c)); }
    });
    html += legendHtml([
      {cls:"q", text:"position t"},
      {cls:"v", text:"nearby keys for this query"},
      {cls:"d", text:"far pairs, smaller score"}
    ]);
    return html;
  }
  function genericPlay() {
    var html = '<div class="mav-strip">';
    for (var i = 0; i < SEQ; i++) html += cell(i, "keep");
    return html + "</div>";
  }
  function playInner(node, opName) {
    var viz = node.viz || "generic";
    if (viz === "attention") return attentionPlay(state.query);
    if (viz === "embedding") return embeddingPlay(state.token);
    if (viz === "mlp") return mlpPlay(state.token);
    if (viz === "moe") return moePlay(state.token);
    if (viz === "deepseek_moe") return deepseekMoePlay(node, state.token);
    if (viz === "mla") return mlaPlay(node, state.query, opName);
    if (viz === "csa2") return csa2Play(node, state.query);
    if (viz === "mhc") return mhcPlay(node);
    if (viz === "engram") return engramPlay(node, state.token);
    if (viz === "dspark") return dsparkPlay(node, state.token);
    if (viz === "mtp") return mtpPlay(state.token);
    if (viz === "norm") return normPlay(state.token);
    if (viz === "residual") return residualPlay(state.token);
    if (viz === "block") return blockPlay(state.token);
    if (viz === "lm_head") return lmPlay(state.token);
    if (viz === "rope") return ropePlay(state.token);
    return genericPlay();
  }
  function playHtml(node, opName) {
    return '<div class="mav-play"><div class="mav-play-title">Interactive view</div>' + playInner(node, opName) + "</div>";
  }

  function bindPlayControls(node, opName) {
    var q = explainBody.querySelector("[data-mav-query]");
    var t = explainBody.querySelector("[data-mav-token]");
    function follow(input, key) {
      input.addEventListener("input", function() {
        state[key] = Number(input.value);
        var readout = input.parentNode && input.parentNode.querySelector(".mav-readout");
        if (readout) readout.textContent = input.value;
        cancelAnimationFrame(playFrame);
        playFrame = requestAnimationFrame(function() {
          playFrame = 0;
          refreshPlay(node, opName);
        });
      });
    }
    if (q) follow(q, "query");
    if (t) follow(t, "token");
  }
  function refreshPlay(node, opName) {
    var play = explainBody.querySelector(".mav-play");
    if (!play) return;
    var kept = play.querySelector(".mav-controls");
    var wrap = document.createElement("div");
    wrap.innerHTML = playHtml(node, opName);
    var next = wrap.firstElementChild;
    if (!kept || !next.querySelector(".mav-controls")) {
      play.replaceWith(next);
      bindPlayControls(node, opName);
      return;
    }
    var child = play.firstChild;
    while (child) {
      var upcoming = child.nextSibling;
      if (child !== kept) play.removeChild(child);
      child = upcoming;
    }
    var incoming = next.firstChild;
    var before = true;
    while (incoming) {
      var upcomingIn = incoming.nextSibling;
      var isControls = incoming.nodeType === 1 && incoming.classList.contains("mav-controls");
      if (isControls) before = false;
      else if (before) play.insertBefore(incoming, kept);
      else play.appendChild(incoming);
      incoming = upcomingIn;
    }
  }

  function renderExplain() {
    cancelAnimationFrame(playFrame);
    playFrame = 0;
    var step = steps[state.step];
    var node = data.nodes[step.node];
    var op = step.opData;
    var nOps = (node.ops && node.ops.length) ? node.ops.length : 1;
    var html = "<h2>" + esc(node.label) + "</h2>";
    html += '<div class="mav-kind">' + esc(node.kind);
    if (node.repeat && node.repeat > 1) html += " · repeated " + node.repeat + " times";
    html += " · operation " + (step.op + 1) + " of " + nOps + " in this block</div>";
    if ((node.inputs && node.inputs.length) || (node.outputs && node.outputs.length)) {
      html += '<div class="mav-shapes">';
      (node.inputs || []).forEach(function(t) {
        html += '<div class="mav-shape"><span>in</span>' + esc(t.name) + " " + esc(shapeText(t.shape)) + "</div>";
      });
      (node.outputs || []).forEach(function(t) {
        html += '<div class="mav-shape"><span>out</span>' + esc(t.name) + " " + esc(shapeText(t.shape)) + "</div>";
      });
      html += "</div>";
    }
    if (nOps > 1) {
      html += '<div class="mav-op-chips">';
      for (var j = 0; j < nOps; j++) {
        var name = node.ops[j] ? node.ops[j].name : node.kind;
        html += '<button type="button" class="mav-op-chip' + (j === step.op ? " is-on" : "") + '" data-op="' + j + '">' + esc(name) + "</button>";
      }
      html += "</div>";
    }
    if (op.math) {
      html += '<div class="mav-mathblock"><div class="mav-op-name">' + esc(op.name) + '</div><div class="mav-op-math">' + esc(op.math) + "</div></div>";
    }
    html += '<p class="mav-prose">' + esc(op.note || "Step through the operations of this block.") + "</p>";
    if (node.meta && node.meta.tensor_shapes) {
      html += '<div class="mav-shapes">';
      Object.keys(node.meta.tensor_shapes).forEach(function(k) {
        html += '<div class="mav-shape"><span>w</span>' + esc(k) + " " + esc(shapeText(node.meta.tensor_shapes[k])) + "</div>";
      });
      html += "</div>";
    }
    html += playHtml(node, op.name);
    explainBody.innerHTML = html;
    explainBody.scrollTop = 0;
    bindPlayControls(node, op.name);

    countEl.innerHTML = '<span class="mav-step-now">' + (state.step + 1) + "</span> / " + steps.length +
      "  ·  " + esc(node.label) + '<span class="mav-keys">arrow keys</span>';
    prevBtn.disabled = state.step === 0;
    nextBtn.disabled = state.step === steps.length - 1;
    var segs = progress.querySelectorAll(".mav-seg");
    segs.forEach(function(seg, i) {
      seg.classList.toggle("is-on", i === state.step);
      if (i === state.step) seg.setAttribute("aria-current", "step");
      else seg.removeAttribute("aria-current");
    });
  }

  function scrollNodeIntoPane(el) {
    var elRect = el.getBoundingClientRect();
    var paneRect = arch.getBoundingClientRect();
    if (elRect.top >= paneRect.top + 36 && elRect.bottom <= paneRect.bottom - 8) return;
    var delta = elRect.top < paneRect.top + 36
      ? elRect.top - paneRect.top - 36
      : elRect.bottom - paneRect.bottom + 8;
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    arch.scrollBy({top: delta, behavior: reduce ? "auto" : "smooth"});
  }

  function paintSelection(scroll) {
    var idx = steps[state.step].node;
    arch.querySelectorAll(".mav-node").forEach(function(el) {
      var on = Number(el.getAttribute("data-idx")) === idx;
      el.classList.toggle("is-on", on);
      if (on) el.setAttribute("aria-current", "true");
      else el.removeAttribute("aria-current");
    });
    arch.querySelectorAll(".mav-module").forEach(function(mod) {
      mod.classList.toggle("is-current", !!mod.querySelector(".mav-node.is-on"));
    });
    if (!scroll) return;
    var el = arch.querySelector('.mav-node[data-idx="' + idx + '"]');
    if (el) scrollNodeIntoPane(el);
  }

  function go(step) {
    var next = Math.max(0, Math.min(steps.length - 1, step));
    if (next === state.step && explainBody.childElementCount) return;
    var prevNode = steps[state.step] ? steps[state.step].node : -1;
    state.step = next;
    paintSelection(steps[next].node !== prevNode);
    renderExplain();
  }

  function buildTree(nodes) {
    var tree = [];
    var stack = [{depth: -1, children: tree}];
    nodes.forEach(function(node, idx) {
      var item = {idx: idx, node: node, children: []};
      while (stack.length > 1 && stack[stack.length - 1].depth >= node.depth) stack.pop();
      stack[stack.length - 1].children.push(item);
      stack.push({depth: node.depth, children: item.children});
    });
    return tree;
  }
  function linkEl() {
    var d = document.createElement("div");
    d.className = "mav-link";
    d.setAttribute("aria-hidden", "true");
    return d;
  }
  function nodeButton(item) {
    var node = item.node;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mav-node viz-" + safeClass(node.viz);
    btn.setAttribute("data-idx", String(item.idx));
    var showIo = node.viz === "embedding" || node.viz === "lm_head" || node.viz === "mtp";
    var io = showIo ? ioLine(node) : "";
    var repeat = node.repeat && node.repeat > 1 ? ' <span class="mav-repeat">× ' + node.repeat + "</span>" : "";
    var kind = String(node.kind || "").replace(/_/g, " ");
    btn.innerHTML = '<span class="mav-kicker">' + esc(kind) + "</span>" +
      '<span class="mav-name">' + esc(node.label) + repeat + "</span>" +
      (io ? '<span class="mav-io" title="' + esc(io) + '">' + esc(io) + "</span>" : "");
    return btn;
  }
  function renderItem(item) {
    if (!item.children.length) return nodeButton(item);
    var mod = document.createElement("div");
    mod.className = "mav-module" + (item.node.repeat > 1 ? " is-deck" : "");
    var body = document.createElement("div");
    body.className = "mav-module-body";
    body.appendChild(nodeButton(item));
    var nest = document.createElement("div");
    nest.className = "mav-nest";
    nest.appendChild(linkEl());
    renderItems(item.children, nest);
    body.appendChild(nest);
    mod.appendChild(body);
    return mod;
  }
  function renderItems(items, host) {
    items.forEach(function(item, i) {
      if (i > 0) host.appendChild(linkEl());
      host.appendChild(renderItem(item));
    });
  }

  renderItems(buildTree(data.nodes), diagram);

  var seen = [];
  var vizName = {
    embedding: "Embedding", attention: "Attention", mla: "Latent attention",
    csa2: "Compressed attention", mhc: "Hyper-connection",
    norm: "Norm", mlp: "MLP", moe: "MoE", deepseek_moe: "DeepSeekMoE",
    engram: "Engram", residual: "Residual", block: "Repeated layer",
    lm_head: "LM head", rope: "RoPE", mtp: "Multi-token",
    dspark: "DSpark", generic: "Other"
  };
  data.nodes.forEach(function(n) {
    var v = safeClass(n.viz);
    if (seen.indexOf(v) === -1) seen.push(v);
  });
  seen.forEach(function(v) {
    var s = document.createElement("span");
    s.className = "mav-legend-item";
    s.innerHTML = '<i class="viz-' + v + '"></i>' + esc(vizName[v] || v);
    legend.appendChild(s);
  });

  steps.forEach(function(step, i) {
    var node = data.nodes[step.node];
    var seg = document.createElement("button");
    seg.type = "button";
    seg.className = "mav-seg viz-" + safeClass(node.viz);
    seg.setAttribute("data-step", String(i));
    seg.title = node.label + " · " + (step.opData.name || "");
    progress.appendChild(seg);
  });

  arch.addEventListener("click", function(ev) {
    var btn = ev.target.closest(".mav-node");
    if (!btn) return;
    var idx = Number(btn.getAttribute("data-idx"));
    for (var i = 0; i < steps.length; i++) {
      if (steps[i].node === idx) { go(i); return; }
    }
  });
  explainBody.addEventListener("click", function(ev) {
    var chip = ev.target.closest("[data-op]");
    if (!chip) return;
    var want = Number(chip.getAttribute("data-op"));
    var nodeIdx = steps[state.step].node;
    for (var i = 0; i < steps.length; i++) {
      if (steps[i].node === nodeIdx && steps[i].op === want) { go(i); return; }
    }
  });
  prevBtn.addEventListener("click", function() { go(state.step - 1); });
  nextBtn.addEventListener("click", function() { go(state.step + 1); });
  progress.addEventListener("click", function(ev) {
    var seg = ev.target.closest("[data-step]");
    if (!seg) return;
    go(Number(seg.getAttribute("data-step")));
  });
  root.addEventListener("keydown", function(e) {
    if (e.target && e.target.matches && e.target.matches("input, textarea, select")) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      go(state.step + 1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      go(state.step - 1);
    }
  });

  go(0);
})();
  </script>
</div>
""".strip()
