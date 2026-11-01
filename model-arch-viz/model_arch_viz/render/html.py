"""Self-contained HTML architecture widget with interactive math strips."""

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
    body = _shell(arch, nodes)
    if standalone:
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
            f"<title>{html.escape(arch.name)}</title>\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
            "</head>\n<body style=\"margin:0;background:#f4f2ee;\">\n"
            f"{body}\n</body>\n</html>\n"
        )
    return body


def _flatten(arch: Architecture) -> list[dict]:
    rows: list[dict] = []

    def walk(layer: Layer, depth: int) -> None:
        label = layer.name
        if layer.repeat and layer.repeat > 1:
            label = f"{layer.name} × {layer.repeat}"
        rows.append(
            {
                "id": layer.id,
                "label": label,
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


def _shell(arch: Architecture, nodes: list[dict]) -> str:
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

    return f"""
<div id="{uid}" class="mav-root" data-mav="1">
<style>
#{uid}.mav-root {{
  --mav-bg: #f7f5f1;
  --mav-panel: #fffcf8;
  --mav-ink: #1c1917;
  --mav-muted: #78716c;
  --mav-line: #e7e5e4;
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
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--mav-ink);
  background: var(--mav-bg);
  border: 1px solid var(--mav-line);
  border-radius: 14px;
  overflow: hidden;
  max-width: 980px;
  margin: 0 auto;
  font-weight: 400;
}}
#{uid} * {{ box-sizing: border-box; font-weight: inherit; }}
#{uid} b, #{uid} strong {{ font-weight: 400; }}
#{uid} .mav-header {{
  padding: 18px 20px 14px;
  border-bottom: 1px solid var(--mav-line);
  background: var(--mav-panel);
}}
#{uid} .mav-title {{
  margin: 0;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
}}
#{uid} .mav-sub {{
  margin: 4px 0 0;
  color: var(--mav-muted);
  font-size: 0.85rem;
}}
#{uid} .mav-chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}}
#{uid} .mav-chip {{
  font-size: 0.75rem;
  color: var(--mav-muted);
  background: var(--mav-bg);
  border: 1px solid var(--mav-line);
  border-radius: 999px;
  padding: 3px 10px;
}}
#{uid} .mav-sym {{ color: var(--mav-accent); margin-right: 2px; }}
#{uid} .mav-body {{
  display: grid;
  grid-template-columns: minmax(190px, 30%) 1fr;
  min-height: 420px;
}}
@media (max-width: 640px) {{
  #{uid} .mav-body {{ grid-template-columns: 1fr; }}
}}
#{uid} .mav-list {{
  border-right: 1px solid var(--mav-line);
  background: var(--mav-panel);
  overflow: auto;
  max-height: 640px;
  padding: 8px;
}}
#{uid} .mav-item {{
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  cursor: pointer;
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 2px;
}}
#{uid} .mav-item:hover {{ background: var(--mav-bg); }}
#{uid} .mav-item[aria-selected="true"] {{
  background: var(--mav-accent-soft);
  color: var(--mav-select);
}}
#{uid} .mav-item-kind {{
  display: block;
  font-size: 0.7rem;
  color: var(--mav-muted);
  margin-top: 2px;
}}
#{uid} .mav-detail {{
  padding: 16px 18px 20px;
  overflow: auto;
  max-height: 640px;
  background: var(--mav-bg);
}}
#{uid} .mav-detail h2 {{
  margin: 0 0 4px;
  font-size: 1.05rem;
}}
#{uid} .mav-kind {{
  color: var(--mav-muted);
  font-size: 0.8rem;
  margin-bottom: 12px;
}}
#{uid} .mav-section {{ margin-top: 14px; }}
#{uid} .mav-section-label {{
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--mav-muted);
  margin-bottom: 8px;
}}
#{uid} .mav-shapes {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}
#{uid} .mav-shape {{
  background: var(--mav-panel);
  border: 1px solid var(--mav-line);
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 0.85rem;
}}
#{uid} .mav-shape span {{ color: var(--mav-muted); margin-right: 6px; }}
#{uid} .mav-op {{
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
  gap: 10px 14px;
  background: var(--mav-panel);
  border: 1px solid var(--mav-line);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 8px;
}}
@media (max-width: 640px) {{
  #{uid} .mav-op {{ grid-template-columns: 1fr; }}
}}
#{uid} .mav-op-name {{
  grid-column: 1 / -1;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--mav-accent);
  margin-bottom: 2px;
}}
#{uid} .mav-op-math {{
  font-family: "Cambria Math", "Times New Roman", Times, serif;
  font-size: 1.02rem;
  line-height: 1.45;
  white-space: pre-wrap;
}}
#{uid} .mav-op-note {{
  font-size: 0.84rem;
  line-height: 1.45;
  color: var(--mav-muted);
}}
#{uid} .mav-play {{
  margin-top: 16px;
  background: var(--mav-panel);
  border: 1px solid var(--mav-line);
  border-radius: 12px;
  padding: 12px 14px 14px;
}}
#{uid} .mav-play-title {{
  font-size: 0.72rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--mav-muted);
  margin-bottom: 8px;
}}
#{uid} .mav-play-formula {{
  font-family: "Cambria Math", "Times New Roman", Times, serif;
  font-size: 0.95rem;
  margin-bottom: 10px;
}}
#{uid} .mav-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px 18px;
  align-items: center;
  margin-bottom: 12px;
}}
#{uid} .mav-controls label {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.85rem;
  color: var(--mav-muted);
}}
#{uid} .mav-controls input[type="range"] {{
  width: 160px;
  accent-color: var(--mav-accent);
}}
#{uid} .mav-readout {{
  color: var(--mav-ink);
  min-width: 2ch;
}}
#{uid} .mav-strip {{
  display: flex;
  flex-wrap: nowrap;
  gap: 3px;
  overflow-x: auto;
  padding-bottom: 4px;
}}
#{uid} .mav-cell {{
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border-radius: 5px;
  border: 1px solid var(--mav-line);
  background: var(--mav-drop);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  color: var(--mav-muted);
}}
#{uid} .mav-cell.role-query {{ background: var(--mav-purple-soft); border-color: var(--mav-purple); color: var(--mav-purple); }}
#{uid} .mav-cell.role-visible {{ background: var(--mav-green-soft); border-color: var(--mav-green); color: var(--mav-green); }}
#{uid} .mav-cell.role-keep {{ background: var(--mav-blue-soft); border-color: var(--mav-blue); color: var(--mav-blue); }}
#{uid} .mav-cell.role-active {{ background: var(--mav-accent-soft); border-color: var(--mav-accent); color: var(--mav-select); }}
#{uid} .mav-cell.role-gate {{ background: var(--mav-amber-soft); border-color: var(--mav-amber); color: var(--mav-amber); }}
#{uid} .mav-cell.role-dropped {{ background: var(--mav-drop); color: #a8a29e; }}
#{uid} .mav-brackets {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
  font-size: 0.75rem;
  color: var(--mav-muted);
}}
#{uid} .mav-bracket span {{
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 5px;
  vertical-align: -1px;
  border: 1px solid var(--mav-line);
}}
#{uid} .mav-bracket .q {{ background: var(--mav-purple-soft); border-color: var(--mav-purple); }}
#{uid} .mav-bracket .v {{ background: var(--mav-green-soft); border-color: var(--mav-green); }}
#{uid} .mav-bracket .k {{ background: var(--mav-blue-soft); border-color: var(--mav-blue); }}
#{uid} .mav-bracket .a {{ background: var(--mav-accent-soft); border-color: var(--mav-accent); }}
#{uid} .mav-bracket .g {{ background: var(--mav-amber-soft); border-color: var(--mav-amber); }}
#{uid} .mav-bracket .d {{ background: var(--mav-drop); }}
#{uid} .mav-matrix {{
  margin-top: 10px;
  overflow-x: auto;
}}
#{uid} .mav-matrix table {{
  border-collapse: collapse;
  font-size: 0.72rem;
}}
#{uid} .mav-matrix th, #{uid} .mav-matrix td {{
  width: 22px;
  height: 22px;
  text-align: center;
  border: 1px solid var(--mav-line);
  color: var(--mav-muted);
}}
#{uid} .mav-matrix td.on {{
  background: var(--mav-green-soft);
  color: var(--mav-green);
}}
#{uid} .mav-matrix td.query-row {{ outline: 1px solid var(--mav-purple); }}
#{uid} .mav-notes {{
  margin: 12px 20px 16px;
  padding: 0;
  color: var(--mav-muted);
  font-size: 0.8rem;
}}
#{uid} .mav-notes li {{ margin-bottom: 4px; }}
#{uid} .mav-empty {{ padding: 24px; color: var(--mav-muted); }}
@media (prefers-reduced-motion: reduce) {{
  #{uid} .mav-cell {{ transition: none; }}
}}
</style>
  <div class="mav-header">
    <h1 class="mav-title">{html.escape(arch.name)}</h1>
    <p class="mav-sub">{html.escape(arch.source or "architecture")}</p>
    <div class="mav-chips">{dims_html}</div>
  </div>
  <div class="mav-body">
    <div class="mav-list" role="listbox" aria-label="Layers"></div>
    <div class="mav-detail" aria-live="polite"></div>
  </div>
  {"<ul class='mav-notes'>" + notes_html + "</ul>" if notes_html else ""}
  <script type="application/json" id="{uid}-data">{data_json}</script>
  <script>
(function() {{
  var root = document.getElementById("{uid}");
  if (!root) return;
  var data = JSON.parse(document.getElementById("{uid}-data").textContent);
  var list = root.querySelector(".mav-list");
  var detail = root.querySelector(".mav-detail");
  if (!data.nodes || !data.nodes.length) {{
    detail.innerHTML = '<p class="mav-empty">No layers in this architecture.</p>';
    return;
  }}

  var SEQ = 12;
  var state = {{ idx: 0, query: 7, token: 3 }};

  function esc(s) {{
    return String(s == null ? "" : s).replace(/[&<>"']/g, function(c) {{
      return ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[c];
    }});
  }}

  function shapeText(arr) {{
    return "[" + (arr || []).join(", ") + "]";
  }}

  function cell(i, role, label) {{
    return '<div class="mav-cell role-' + role + '" title="' + esc(String(label != null ? label : i)) + '">' + esc(String(i)) + '</div>';
  }}

  function legend(items) {{
    return '<div class="mav-brackets">' + items.map(function(it) {{
      return '<span class="mav-bracket"><span class="' + it.cls + '"></span>' + esc(it.text) + '</span>';
    }}).join("") + '</div>';
  }}

  function attentionPlay(query) {{
    var html = '<div class="mav-play-formula">M[i, j] = 1 iff j ≤ i (causal). Softmax over visible keys only.</div>';
    html += '<div class="mav-controls"><label>query i <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + query + '" data-mav-query/><span class="mav-readout" data-mav-qval>' + query + '</span></label></div>';
    html += '<div class="mav-strip" aria-label="Key positions for query">';
    for (var j = 0; j < SEQ; j++) {{
      var role = "dropped";
      if (j === query) role = "query";
      else if (j <= query) role = "visible";
      html += cell(j, role, "key " + j);
    }}
    html += '</div>';
    html += legend([
      {{cls:"q", text:"query i"}},
      {{cls:"v", text:"visible keys [0, i]"}},
      {{cls:"d", text:"masked future j > i"}}
    ]);
    html += '<div class="mav-matrix"><table aria-label="Causal mask"><thead><tr><th></th>';
    for (var c = 0; c < SEQ; c++) html += '<th>' + c + '</th>';
    html += '</tr></thead><tbody>';
    for (var r = 0; r < SEQ; r++) {{
      html += '<tr' + (r === query ? ' class="query-row"' : '') + '><th>' + r + '</th>';
      for (var c2 = 0; c2 < SEQ; c2++) {{
        var on = c2 <= r;
        html += '<td class="' + (on ? 'on' : '') + (r === query ? ' query-row' : '') + '">' + (on ? '1' : '0') + '</td>';
      }}
      html += '</tr>';
    }}
    html += '</tbody></table></div>';
    return html;
  }}

  function embeddingPlay(token) {{
    var html = '<div class="mav-play-formula">X[t] = Embed[token_ids[t]] · row lookup in table of size V × H</div>';
    html += '<div class="mav-controls"><label>token t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-strip" aria-label="Sequence tokens">';
    for (var t = 0; t < SEQ; t++) {{
      html += cell(t, t === token ? "active" : "dropped", "token " + t);
    }}
    html += '</div>';
    html += legend([
      {{cls:"a", text:"selected token t"}},
      {{cls:"d", text:"other positions"}}
    ]);
    html += '<div class="mav-play-formula" style="margin-top:10px">table row for token_ids[' + token + '] → vector of length H</div>';
    html += '<div class="mav-strip">';
    for (var d = 0; d < 10; d++) html += cell(d, "keep", "dim " + d);
    html += '<div class="mav-cell role-dropped">…</div></div>';
    return html;
  }}

  function mlpPlay(token) {{
    var html = '<div class="mav-play-formula">H → I (expand) → act → H (contract) at each token</div>';
    html += '<div class="mav-controls"><label>token t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-section-label">sequence</div><div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) html += cell(t, t === token ? "query" : "dropped");
    html += '</div>';
    html += '<div class="mav-section-label" style="margin-top:10px">width at token t</div>';
    html += '<div class="mav-strip">';
    for (var a = 0; a < 4; a++) html += cell("H", "keep");
    html += '<div class="mav-cell role-dropped">→</div>';
    for (var b = 0; b < 8; b++) html += cell("I", "gate");
    html += '<div class="mav-cell role-dropped">→</div>';
    for (var c = 0; c < 4; c++) html += cell("H", "active");
    html += '</div>';
    html += legend([
      {{cls:"q", text:"token t"}},
      {{cls:"k", text:"hidden H"}},
      {{cls:"g", text:"intermediate I"}},
      {{cls:"a", text:"output H"}}
    ]);
    return html;
  }}

  function moePlay(token) {{
    var E = 8, topk = 2;
    var scores = [];
    for (var e = 0; e < E; e++) scores.push({{e:e, s:(Math.sin(token * 1.7 + e) + 1) / 2}});
    scores.sort(function(a,b){{ return b.s - a.s; }});
    var chosen = {{}};
    for (var k = 0; k < topk; k++) chosen[scores[k].e] = true;
    var html = '<div class="mav-play-formula">route token t to top-' + topk + ' of ' + E + ' experts; Y = Σ p_e Y_e</div>';
    html += '<div class="mav-controls"><label>token t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-strip" aria-label="Experts">';
    for (var i = 0; i < E; i++) html += cell(i, chosen[i] ? "active" : "dropped", "expert " + i);
    html += '</div>';
    html += legend([
      {{cls:"a", text:"top-k experts"}},
      {{cls:"d", text:"idle experts"}}
    ]);
    return html;
  }}

  function normPlay(token) {{
    var html = '<div class="mav-play-formula">features of token t are rescaled together (RMS or mean/var)</div>';
    html += '<div class="mav-controls"><label>token t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) html += cell(t, t === token ? "active" : "dropped");
    html += '</div>';
    html += '<div class="mav-section-label" style="margin-top:10px">feature axis at t</div><div class="mav-strip">';
    for (var d = 0; d < 12; d++) html += cell(d, "keep");
    html += '</div>';
    html += legend([{{cls:"a", text:"token t"}}, {{cls:"k", text:"features normalized together"}}]);
    return html;
  }}

  function residualPlay(token) {{
    var html = '<div class="mav-play-formula">Y[t] = X[t] + F(X)[t]</div>';
    html += '<div class="mav-controls"><label>token t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) {{
      var role = "dropped";
      if (t === token) role = "query";
      html += cell(t, role);
    }}
    html += '</div>';
    html += legend([
      {{cls:"q", text:"X[t] kept"}},
      {{cls:"a", text:"F(X)[t] added at same index"}}
    ]);
    return html;
  }}

  function blockPlay(query) {{
    return attentionPlay(query) + '<div style="height:12px"></div><div class="mav-play-formula">Then MLP/MoE runs per token with a residual add — same t, no mixing across sequence in the FFN.</div>';
  }}

  function lmPlay(token) {{
    var html = '<div class="mav-play-formula">logits[t, v] = X[t] · W_lm[:, v]</div>';
    html += '<div class="mav-controls"><label>position t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) html += cell(t, t === token ? "query" : "dropped");
    html += '</div>';
    html += '<div class="mav-section-label" style="margin-top:10px">vocab scores at t (illustrative)</div><div class="mav-strip">';
    for (var v = 0; v < 16; v++) {{
      var hot = (v === (token * 3) % 16) || (v === (token * 3 + 1) % 16);
      html += cell(v, hot ? "active" : "keep");
    }}
    html += '</div>';
    html += legend([{{cls:"q", text:"position t"}}, {{cls:"a", text:"higher logit bins"}}, {{cls:"k", text:"other vocab"}}]);
    return html;
  }}

  function ropePlay(token) {{
    var html = '<div class="mav-play-formula">(Q′, K′) rotate by θ · position; relative offset enters Q·K</div>';
    html += '<div class="mav-controls"><label>position t <input type="range" min="0" max="' + (SEQ - 1) + '" value="' + token + '" data-mav-token/><span class="mav-readout" data-mav-tval>' + token + '</span></label></div>';
    html += '<div class="mav-strip">';
    for (var t = 0; t < SEQ; t++) {{
      var role = "dropped";
      if (t === token) role = "query";
      else if (Math.abs(t - token) <= 2) role = "visible";
      html += cell(t, role);
    }}
    html += '</div>';
    html += legend([{{cls:"q", text:"position t"}}, {{cls:"v", text:"nearby positions (relative)"}}]);
    return html;
  }}

  function genericPlay() {{
    return '<div class="mav-play-formula">Follow the formulas above; strip follows tensor index roles for this module.</div>' +
      '<div class="mav-strip">' + Array.from({{length: SEQ}}, function(_, i){{ return cell(i, "keep"); }}).join("") + '</div>';
  }}

  function playHtml(node) {{
    var viz = node.viz || "generic";
    var inner = "";
    if (viz === "attention") inner = attentionPlay(state.query);
    else if (viz === "embedding") inner = embeddingPlay(state.token);
    else if (viz === "mlp") inner = mlpPlay(state.token);
    else if (viz === "moe") inner = moePlay(state.token);
    else if (viz === "norm") inner = normPlay(state.token);
    else if (viz === "residual") inner = residualPlay(state.token);
    else if (viz === "block") inner = blockPlay(state.query);
    else if (viz === "lm_head") inner = lmPlay(state.token);
    else if (viz === "rope") inner = ropePlay(state.token);
    else inner = genericPlay();
    return '<div class="mav-play"><div class="mav-play-title">Interactive view</div>' + inner + '</div>';
  }}

  function bindPlayControls(node) {{
    var q = detail.querySelector("[data-mav-query]");
    var t = detail.querySelector("[data-mav-token]");
    if (q) q.addEventListener("input", function() {{
      state.query = Number(q.value);
      var ro = detail.querySelector("[data-mav-qval]");
      if (ro) ro.textContent = String(state.query);
      refreshPlay(node);
    }});
    if (t) t.addEventListener("input", function() {{
      state.token = Number(t.value);
      var ro = detail.querySelector("[data-mav-tval]");
      if (ro) ro.textContent = String(state.token);
      refreshPlay(node);
    }});
  }}

  function refreshPlay(node) {{
    var play = detail.querySelector(".mav-play");
    if (!play) return;
    var wrap = document.createElement("div");
    wrap.innerHTML = playHtml(node);
    play.replaceWith(wrap.firstChild);
    bindPlayControls(node);
  }}

  function renderDetail(node) {{
    var html = "<h2>" + esc(node.label) + "</h2>";
    html += '<div class="mav-kind">' + esc(node.kind) + (node.repeat ? " · repeat " + node.repeat : "") + "</div>";

    if ((node.inputs && node.inputs.length) || (node.outputs && node.outputs.length)) {{
      html += '<div class="mav-section"><div class="mav-section-label">Tensors</div><div class="mav-shapes">';
      (node.inputs || []).forEach(function(t) {{
        html += '<div class="mav-shape"><span>in</span>' + esc(t.name) + " " + esc(shapeText(t.shape)) + "</div>";
      }});
      (node.outputs || []).forEach(function(t) {{
        html += '<div class="mav-shape"><span>out</span>' + esc(t.name) + " " + esc(shapeText(t.shape)) + "</div>";
      }});
      html += "</div></div>";
    }}

    if (node.ops && node.ops.length) {{
      html += '<div class="mav-section"><div class="mav-section-label">Operations</div>';
      node.ops.forEach(function(op) {{
        html += '<div class="mav-op"><div class="mav-op-name">' + esc(op.name) + "</div>";
        html += '<div class="mav-op-math">' + esc(op.math) + "</div>";
        html += '<div class="mav-op-note">' + esc(op.note || "See the interactive strip for how indices move under this step.") + "</div>";
        html += "</div>";
      }});
      html += "</div>";
    }}

    if (node.meta && node.meta.tensor_shapes) {{
      html += '<div class="mav-section"><div class="mav-section-label">Weight shapes</div><div class="mav-shapes">';
      Object.keys(node.meta.tensor_shapes).forEach(function(k) {{
        html += '<div class="mav-shape"><span>w</span>' + esc(k) + " " + esc(shapeText(node.meta.tensor_shapes[k])) + "</div>";
      }});
      html += "</div></div>";
    }}

    html += playHtml(node);
    detail.innerHTML = html;
    bindPlayControls(node);
  }}

  function select(idx) {{
    state.idx = idx;
    var buttons = list.querySelectorAll(".mav-item");
    buttons.forEach(function(btn, i) {{
      btn.setAttribute("aria-selected", i === idx ? "true" : "false");
    }});
    renderDetail(data.nodes[idx]);
  }}

  data.nodes.forEach(function(node, idx) {{
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mav-item";
    btn.setAttribute("role", "option");
    btn.style.paddingLeft = (10 + (node.depth || 0) * 14) + "px";
    btn.innerHTML = esc(node.label) + '<span class="mav-item-kind">' + esc(node.kind) + "</span>";
    btn.addEventListener("click", function() {{ select(idx); }});
    list.appendChild(btn);
  }});

  select(0);
}})();
  </script>
</div>
""".strip()
