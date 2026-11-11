#!/usr/bin/env python3
"""Build a chemiscope explorer: several pre-configured views over the scored structures, in one page.

    python tools/build_chemiscope_views.py RUNS_DIR [--data DIR] [-o OUT.html] [--open]

No model is re-evaluated. Every number comes from the canonical records and the per-term energies that
``lrbench eval --save-predictions`` already wrote, recombined through each group's own definition, so a model value
and its reference are formed identically.

Each view is a separate chemiscope instance with its own default axes, colouring and question. They share one
embedded dataset per set, so the page costs one copy of the data no matter how many views look at it, and every
view keeps chemiscope's own axis, colour, size and atom-colour controls -- the point is to hand those controls over
already pointed at something worth looking at, not to wrap them in menus of my own.

Global (per-structure) properties: class, separation, reference E_int, and per model its E_int, signed error and
absolute error. Atomic properties: fragment id, formal charge, and the DFT Löwdin charge, with an environment on
every atom so the structure panel can colour by them.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import pathlib
import sys
import webbrowser

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

MEV = 1.0e3
PATHOLOGY = "dft_pathology_do_not_score"


# ------------------------------------------------------------------------------------------------- the data layer
def load_set(data: str, name: str):
    """Records and E_int/E3/ddE groups of one canonical set."""
    jl = os.path.join(data, "canonical", f"{name}.jsonl")
    gp = os.path.join(data, "canonical", f"{name}.groups.json")
    if not os.path.exists(jl):
        return [], []
    with open(jl) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    groups = []
    if os.path.exists(gp):
        with open(gp) as f:
            groups = json.load(f)
    return recs, groups


def group_value(g: dict, preds: dict):
    """The group's linear combination of predicted terms, or None if any term is missing."""
    total = 0.0
    for term, coef in g.get("terms") or []:
        vals = preds.get(term)
        if not vals:
            return None
        good = [v for v in vals if isinstance(v, (int, float))]
        if not good:
            return None
        total += float(coef) * (sum(good) / len(good))
    return total


def entries(results: dict, recs: list, setname: str, groups: list, kind: str):
    """One entry per scored group: the structure its first term names, the reference, and each model's value.

    Keying on the group rather than the record matters for ddE and E3, where several groups share a first term --
    keying on the record id would silently keep one of them and drop the rest.
    """
    by_id = {r["id"]: r for r in recs}
    gs = [g for g in groups if g.get("kind") == kind and g.get("ref") is not None
          and not any(PATHOLOGY in str(x) for x in ((g.get("meta") or {}).get("flags") or []))]
    preds = {name: (((J.get("sets") or {}).get(setname) or {}).get("predictions") or {})
             for name, J in results.items()}
    preds = {k: v for k, v in preds.items() if v}
    out = []
    for g in gs:
        rid = str((g.get("terms") or [[None]])[0][0] or "").split("#")[0]
        rec = by_id.get(rid)
        if rec is None or any(PATHOLOGY in str(f) for f in (rec.get("flags") or [])):
            continue
        vals = {}
        for name, P in preds.items():
            v = group_value(g, P)
            if v is not None:
                vals[name] = MEV * v
        if vals:
            out.append({"rec": rec, "ref": MEV * float(g["ref"]), "vals": vals,
                        "meta": g.get("meta") or {}, "name": g.get("name", "")})
    return out, sorted({m for e in out for m in e["vals"]})


def _nullify(x):
    """NaN is not valid JSON, and a browser's JSON.parse refuses it: write null and let the page restore NaN."""
    if isinstance(x, float):
        return None if (x != x or x in (float("inf"), float("-inf"))) else x
    if isinstance(x, dict):
        return {k: _nullify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nullify(v) for v in x]
    return x


def encode(payload: dict) -> str:
    return base64.b64encode(gzip.compress(
        json.dumps(_nullify(payload), allow_nan=False).encode())).decode()


def build_dataset(ents, models, setname, kind_label):
    """A chemiscope dataset: structures, global properties, atomic properties, an environment on every atom."""
    from ase import Atoms

    if not ents:
        return None
    keep = [e["rec"] for e in ents]
    frames, env = [], []
    for i, r in enumerate(keep):
        a = Atoms(r["species"], positions=np.asarray(r["positions"], float))
        if r.get("cell") is not None:
            a.set_cell(np.asarray(r["cell"], float))
            a.set_pbc(bool(np.any(r.get("pbc") or False)))
        frames.append(a)
        env += [(i, j, 3.5) for j in range(len(a))]

    def num(vals, unit, desc):
        return {"target": "structure", "values": [float(v) if v is not None else float("nan") for v in vals],
                "units": unit, "description": desc}

    def cat(vals, desc):
        return {"target": "structure", "values": [str(v) for v in vals], "description": desc}

    props = {
        "class": cat([r.get("cls") or "?" for r in keep], "interaction class of the pair"),
        "separation": num([r.get("distance") for r in keep], "Å",
                          "closest fragment–fragment contact, the canonical d_min"),
        f"{kind_label}/REF": num([e["ref"] for e in ents], "meV",
                                 f"reference {kind_label} (ωB97M-V/def2-TZVPD unless the set says otherwise)"),
        "total charge": num([r.get("charge") for r in keep], "e", "total charge of the structure"),
        "atoms": num([len(r["species"]) for r in keep], "", "number of atoms"),
    }
    for m in models:
        have = [e["vals"].get(m) for e in ents]
        props[f"{kind_label}/{m}"] = num(have, "meV", f"{m}: {kind_label}")
        props[f"err/{m}"] = num([None if v is None else v - e["ref"] for v, e in zip(have, ents)], "meV",
                                f"{m}: {kind_label} minus reference (signed)")
        props[f"abserr/{m}"] = num([None if v is None else abs(v - e["ref"]) for v, e in zip(have, ents)],
                                   "meV", f"{m}: absolute {kind_label} error")

    # ---- atomic properties: one value per atom of every kept frame, in frame order
    def atomwise(getter, unit, desc):
        vals = []
        for r in keep:
            n = len(r["species"])
            v = getter(r)
            if v is None:
                vals += [float("nan")] * n
            else:
                v = list(v) + [float("nan")] * max(0, n - len(v))
                vals += [float(x) if isinstance(x, (int, float)) else float("nan") for x in v[:n]]
        return {"target": "atom", "values": vals, "units": unit, "description": desc}

    # chemiscope plots environments when a dataset has them, and then only atom properties are valid map axes.
    # So the global and the atomic views get their own dataset over the same structures, rather than one dataset
    # that can only answer half the questions.
    atoms = {
        "fragment": atomwise(lambda r: r.get("fragment_id"), "", "which fragment the atom belongs to"),
        "q_formal": atomwise(lambda r: r.get("formal_charges"), "e", "formal charge assigned to the atom"),
        "q_lowdin": atomwise(lambda r: (r.get("refs") or {}).get("charges_lowdin"), "e",
                             "DFT Löwdin partial charge — the reference for any model's charges"),
    }
    def has_data(v):
        if v["target"] == "atom" or all(isinstance(x, (int, float)) for x in v["values"]):
            vals = [x for x in v["values"] if isinstance(x, (int, float))]
            return bool(vals) and not all(np.isnan(vals))
        return True

    atoms = {k: v for k, v in atoms.items() if has_data(v)}
    props = {k: v for k, v in props.items() if has_data(v)}
    return {"frames": frames, "properties": props, "atom_properties": atoms, "environments": env,
            "n": len(keep), "models": models}


# ------------------------------------------------------------------------------------------------- the page
HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Long-range explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chemiscope@{ver}/dist/chemiscope.min.js"></script>
<style>
:root {{
  --bg:#fcfcfb; --fg:#161616; --mut:#6e6e6e; --rule:#e5e4e1; --card:#ffffff;
  --blue:#0072B2; --verm:#D55E00; --accent:var(--blue);
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --serif:"Crimson Pro",Crimson,Charter,Georgia,serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#131518; --fg:#ebebe8; --mut:#9fa0a0; --rule:#2a2d32; --card:#1a1d21; }} }}
:root[data-theme="dark"] {{ --bg:#131518; --fg:#ebebe8; --mut:#9fa0a0; --rule:#2a2d32; --card:#1a1d21; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ background:var(--bg); color:var(--fg); margin:0; font-family:var(--serif);
  font-size:18px; line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:0 28px; }}
nav.top {{ position:sticky; top:0; z-index:50; background:color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter:blur(10px); border-bottom:1px solid var(--rule); }}
nav.top .wrap {{ display:flex; align-items:center; gap:22px; height:54px; }}
nav.top b {{ font-family:var(--sans); font-size:15px; font-weight:600; letter-spacing:-0.01em; }}
nav.top a {{ font-family:var(--sans); font-size:13.5px; color:var(--mut); text-decoration:none; }}
nav.top a:hover {{ color:var(--fg); }}
header {{ padding:64px 0 10px; }}
h1 {{ font-size:46px; font-weight:600; margin:0 0 14px; letter-spacing:-0.02em; line-height:1.1; }}
h2 {{ font-size:27px; font-weight:600; margin:0 0 6px; letter-spacing:-0.01em; }}
.lede {{ font-size:21px; max-width:68ch; color:var(--fg); }}
p {{ max-width:72ch; }}
.sub, .meta {{ color:var(--mut); font-size:16px; max-width:76ch; }}
.key {{ display:flex; gap:28px; flex-wrap:wrap; margin:26px 0 8px; font-family:var(--sans); }}
.key div {{ border-left:3px solid var(--rule); padding-left:12px; }}
.key b {{ display:block; font-size:13px; font-weight:600; color:var(--mut); text-transform:uppercase;
  letter-spacing:0.06em; }}
.key span {{ font-size:21px; }}
.ok {{ color:var(--blue); }} .bad {{ color:var(--verm); }}
section {{ padding:40px 0 10px; border-top:1px solid var(--rule); margin-top:40px; }}
table.rank {{ border-collapse:collapse; width:100%; font-family:var(--sans); font-size:14px; margin-top:16px; }}
table.rank th {{ text-align:right; padding:8px 10px; border-bottom:1.5px solid var(--rule); cursor:pointer;
  font-weight:600; color:var(--mut); white-space:nowrap; user-select:none; }}
table.rank th:first-child, table.rank td:first-child {{ text-align:left; }}
table.rank th.sorted {{ color:var(--fg); }}
table.rank td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--rule);
  font-variant-numeric:tabular-nums; }}
table.rank tr.ceiling td {{ background:color-mix(in srgb, var(--blue) 8%, transparent); }}
.bar {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:14px 0 10px;
  font-family:var(--sans); font-size:13px; }}
.bar .lbl {{ color:var(--mut); margin-right:2px; font-weight:500; }}
.bar button {{ font-family:var(--sans); font-size:13px; padding:4px 11px; border-radius:999px;
  border:1px solid var(--rule); background:var(--card); color:var(--fg); cursor:pointer; line-height:1.5; }}
.bar button:hover {{ border-color:var(--accent); }}
.bar button[aria-pressed="true"] {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.bar .sep {{ width:1px; height:20px; background:var(--rule); margin:0 6px; }}
.view {{ margin:8px 0 10px; }}
.panes {{ display:flex; flex-wrap:wrap; gap:12px; }}
.left {{ flex:1 1 580px; min-width:320px; }} .right {{ flex:1 1 400px; min-width:290px; }}
.cs-map {{ height:480px; }} .cs-structure {{ height:380px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:14px;
  background:color-mix(in srgb, var(--rule) 55%, transparent); padding:1px 5px; border-radius:4px; }}
.err {{ color:var(--verm); font-family:var(--sans); }}
footer {{ margin:60px 0 70px; color:var(--mut); font-size:15px; }}
@media (max-width:760px) {{ body {{ font-size:17px; }} h1 {{ font-size:34px; }} .wrap {{ padding:0 18px; }}
  .cs-map {{ height:400px; }} nav.top a:nth-child(n+5) {{ display:none; }} }}
</style></head><body>
<nav class="top"><div class="wrap"><b>Long-range explorer</b>
<a href="#ranking">Ranking</a><a href="#v-dist">Error vs distance</a><a href="#v-mm">Model vs model</a>
<a href="#v-atom">Atoms</a><a href="#notes">Notes</a></div></nav>
<div class="wrap">
<header>
<h1>Does this model have<br>any long-range physics?</h1>
<p class="lede">Near equilibrium a potential can reach an excellent force MAE and still be blind past its cutoff.
These diagnostics have a known correct answer, so one number is interpretable on its own.</p>
<div class="key">
  <div><b>tail fraction</b><span><span class="ok">1</span> correct · <span class="bad">0</span> no tail</span></div>
  <div><b>normalised error ε</b><span><span class="ok">0</span> exact · <span class="bad">−1</span> tail missing</span></div>
  <div><b>fixed-charge ΔΔE</b><span><span class="ok">1</span> responds · <span class="bad">0</span> blind</span></div>
</div>
<p class="meta">{summary}</p>
</header>
{ranking}
{sections}
<section id="notes">
<h2>Notes</h2>
<p class="sub">Built from the canonical records and the per-term energies saved by
<code>lrbench eval --save-predictions</code> — no model was re-evaluated. A model's value and its reference are the
same linear combination of predicted terms, so a point compares like with like. Records flagged as DFT pathologies
are excluded. Energies in meV. A model trained at another level of theory is marked ⚠ in the ranking: its energies
are not comparable with these references, its dimensionless diagnostics are.</p>
<p class="sub">The buttons above each view drive chemiscope through its own settings API, so they stay in step with
its panel — open the chemiscope menu for anything the buttons do not cover.</p>
</section>
<footer>lrbench · chemiscope {ver}</footer>
</div>
<script>
const DATA = {data};
const VIEWS = {views};
const RANK = {rank};

async function inflate(b64) {{
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(s).text());
}}
const cache = {{}};
async function dataset(key) {{
  if (!cache[key]) cache[key] = await inflate(DATA[key]);
  const ds = structuredClone(cache[key]);
  for (const p of Object.values(ds.properties))
    if (p.values.every(v => v === null || typeof v === "number"))
      p.values = p.values.map(v => v === null ? NaN : v);
  return ds;
}}

/* ---- the identity line. chemiscope does not expose Plotly, but its map has a relayout of its own; the line is
   computed in data coordinates from the two plotted properties, so it is a true y = x and not a corner-to-corner
   approximation, and it is re-applied whenever a setting changes. ---- */
function diagonalShape(viz, ds) {{
  const s = viz.saveSettings().map;
  const xs = ds.properties[s.x.property], ys = ds.properties[s.y.property];
  if (!xs || !ys) return null;
  const fin = a => a.values.filter(v => typeof v === "number" && isFinite(v));
  const xv = fin(xs), yv = fin(ys);
  if (!xv.length || !yv.length) return null;
  const lo = Math.max(Math.min(...xv), Math.min(...yv));
  const hi = Math.min(Math.max(...xv), Math.max(...yv));
  if (!(hi > lo)) return null;
  return {{ type: "line", xref: "x", yref: "y", x0: lo, y0: lo, x1: hi, y1: hi, layer: "below",
            line: {{ color: "#7a7a7a", width: 1.4, dash: "dot" }} }};
}}
function setDiagonal(st) {{
  try {{
    const shape = st.diag ? diagonalShape(st.viz, st.ds) : null;
    st.viz.map._relayout({{ shapes: shape ? [shape] : [] }});
  }} catch (e) {{ console.warn("diagonal:", e); }}
}}

function button(label, pressed, onclick) {{
  const b = document.createElement("button");
  b.textContent = label;
  b.setAttribute("aria-pressed", pressed ? "true" : "false");
  b.onclick = onclick;
  return b;
}}
function group(bar, label, items, isOn, choose) {{
  if (!items.length) return;
  if (bar.childElementCount) bar.appendChild(Object.assign(document.createElement("span"), {{ className: "sep" }}));
  const l = document.createElement("span");
  l.className = "lbl"; l.textContent = label;
  bar.appendChild(l);
  items.forEach(it => bar.appendChild(button(it.label, isOn(it), () => {{ choose(it); render(bar); }})));
}}
const RENDERERS = new WeakMap();
function render(bar) {{ const f = RENDERERS.get(bar); if (f) f(); }}

function buildBar(st) {{
  const bar = document.getElementById(st.id + "-bar");
  const draw = () => {{
    bar.innerHTML = "";
    const s = st.viz.saveSettings().map;
    const props = Object.keys(st.ds.properties);
    const modelOf = p => {{ const m = /^(?:err|abserr|[^/]+)\\/(.+)$/.exec(p); return m ? m[1] : null; }};
    if (st.models && st.models.length) {{
      const cur = modelOf(s.y.property) || modelOf(s.x.property);
      group(bar, "model", st.models.map(m => ({{ label: m, m }})),
            it => it.m === cur,
            it => {{
              const next = {{}};
              const yk = /^(err|abserr)\\//.exec(s.y.property);
              const xk = /^(err|abserr)\\//.exec(s.x.property);
              if (yk) next.y = {{ property: yk[1] + "/" + it.m }};
              else if (xk) next.x = {{ property: xk[1] + "/" + it.m }};
              else next.y = {{ property: "err/" + it.m }};
              if (st.ds.properties[(next.y || next.x).property]) st.viz.applySettings({{ map: next }});
              setDiagonal(st);
            }});
    }}
    const colours = ["class", "separation", "total charge", "atoms", "fragment", "q_lowdin"]
      .filter(c => props.includes(c))
      .concat(props.filter(p => p.startsWith("abserr/")).slice(0, 2));
    group(bar, "colour", colours.map(c => ({{ label: c, c }})),
          it => it.c === s.color.property,
          it => {{ st.viz.applySettings({{ map: {{ color: {{ property: it.c }} }} }}); }});
    group(bar, "x", [{{ label: "log", v: "log" }}, {{ label: "linear", v: "linear" }}],
          it => it.v === s.x.scale,
          it => {{ st.viz.applySettings({{ map: {{ x: {{ scale: it.v }} }} }}); setDiagonal(st); }});
    group(bar, "", [{{ label: "diagonal", v: 1 }}],
          () => st.diag,
          () => {{ st.diag = !st.diag; setDiagonal(st); }});
  }};
  RENDERERS.set(bar, draw);
  draw();
}}

(async () => {{
  /* ranking table, sortable by clicking a header — no menus */
  const host = document.getElementById("rank-body");
  if (host && RANK.rows.length) {{
    let sortKey = RANK.sort, asc = true;
    const draw = () => {{
      const rows = RANK.rows.slice().sort((a, b) => {{
        if (a.ceiling !== b.ceiling) return a.ceiling ? -1 : 1;
        const x = a[sortKey], y = b[sortKey];
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (asc ? 1 : -1) * (typeof x === "string" ? x.localeCompare(y) : x - y);
      }});
      host.innerHTML = "<thead><tr>" + RANK.cols.map(c =>
        `<th data-k="${{c.k}}" class="${{c.k === sortKey ? "sorted" : ""}}">${{c.t}}</th>`).join("") +
        "</tr></thead><tbody>" + rows.map(r => `<tr class="${{r.ceiling ? "ceiling" : ""}}">` +
          RANK.cols.map(c => `<td>${{r[c.k] === null || r[c.k] === undefined ? "·" : r[c.k]}}</td>`).join("") +
        "</tr>").join("") + "</tbody>";
      host.querySelectorAll("th").forEach(th => th.onclick = () => {{
        const k = th.dataset.k;
        if (k === sortKey) asc = !asc; else {{ sortKey = k; asc = true; }}
        draw();
      }});
    }};
    draw();
  }}

  for (const v of VIEWS) {{
    try {{
      const ds = await dataset(v.key);
      if (v.settings) ds.settings = v.settings;
      const viz = await Chemiscope.DefaultVisualizer.load(
        {{ map: v.id + "-map", info: v.id + "-info", meta: v.id + "-meta", structure: v.id + "-structure" }}, ds);
      const st = {{ id: v.id, viz, ds, models: v.models || [], diag: !!v.diagonal }};
      buildBar(st);
      viz.onSettingChange(() => {{ render(document.getElementById(v.id + "-bar")); }});
      setTimeout(() => setDiagonal(st), 400);
    }} catch (e) {{
      document.getElementById(v.id).innerHTML =
        '<p class="err">chemiscope could not load this view: ' + e + '</p>';
      console.error(e);
    }}
  }}
}})();
</script></body></html>
"""

SECTION = """<section id="{id}">
<h2>{title}</h2>
<p>{prose}</p>
<div class="bar" id="{id}-bar"></div>
<div class="view">
  <div class="panes">
    <div class="left"><div id="{id}-meta"></div><div class="cs-map" id="{id}-map"></div></div>
    <div class="right"><div class="cs-structure" id="{id}-structure"></div><div id="{id}-info"></div></div>
  </div>
</div>
</section>
"""

RANKING = """<section id="ranking">
<h2>The ranking</h2>
<p class="sub">dimers · ion pairs · ordered by |ε|, how much of the physical tail is missing. Click a column to
re-sort. The ceiling row is the dataset's own multipole + induction prior, not a model.</p>
<table class="rank" id="rank-body"></table>
</section>
"""


def payloads(runs_dir, data_dir=None, verbose: bool = True) -> dict:
    """{key: {blob, models, n, label}} — one gzipped, base64 chemiscope dataset per set, plus its atoms twin.

    Shared by the standalone explorer page and by the notebook build, so both show the same thing.
    """
    import chemiscope

    runs = pathlib.Path(runs_dir)
    data = data_dir or str(runs.parent / "lr-sets")
    results = {}
    for p in sorted(runs.glob("*.json")):
        with open(p) as f:
            J = json.load(f)
        if "sets" in J and "oracle" not in p.stem:
            results[p.stem] = J
    if not results:
        return {}

    built = {}
    for setname, kind, label in (("dimers", "E_int", "E_int"), ("trimers", "E3", "E3"),
                                 ("families_fixedq", "ddE", "ddE")):
        recs, groups = load_set(data, setname)
        if not recs:
            continue
        ents, models = entries(results, recs, setname, groups, kind)
        if not ents:
            continue
        ds = build_dataset(ents, models, setname, label)
        if ds is None:
            continue
        payload = chemiscope.create_input(
            structures=ds["frames"], properties=ds["properties"],
            metadata={"name": f"lrbench {setname}",
                      "description": f"{ds['n']} scored {setname} groups, {len(ds['models'])} models"})
        built[setname] = {"blob": encode(payload), "models": ds["models"], "n": ds["n"], "label": label}
        if verbose:
            print(f"[chemiscope] {setname}: {ds['n']} structures, {len(ds['models'])} models, "
                  f"{len(payload['properties'])} global properties")
        if ds["atom_properties"]:
            pay_a = chemiscope.create_input(
                structures=ds["frames"], properties=ds["atom_properties"], environments=ds["environments"],
                metadata={"name": f"lrbench {setname} — atoms",
                          "description": f"every atom of {ds['n']} scored {setname} groups"})
            built[f"{setname}-atoms"] = {"blob": encode(pay_a), "models": ds["models"], "n": ds["n"],
                                         "label": label}
            if verbose:
                print(f"[chemiscope] {setname}-atoms: {len(ds['environments'])} environments, "
                      f"{len(ds['atom_properties'])} atomic properties")
    return built


def pick(models, *want):
    for w in want:
        for m in models:
            if w in m:
                return m
    return models[0] if models else ""


def view_specs(built: dict) -> list:
    """The pre-configured views: (id, dataset key, title, prose, chemiscope settings)."""
    out = []
    dim = built.get("dimers")
    if dim:
        m_lr = pick(dim["models"], "mace-polar", "so3lr-tauri", "orbmol")
        m_loc = pick([m for m in dim["models"] if m != m_lr], "mace-h", "dpa4", "grace", "mace-omol")
        out.append(dict(id="v-dist", key="dimers", title="Where the error lives", prose=(
            f"Separation against {m_lr}'s signed error, coloured by interaction class, with point size carrying "
            f"{m_loc}'s absolute error. Errors that <em>grow</em> with distance are the long-range failure; errors "
            "that shrink with distance are ordinary short-range inaccuracy. Click a point to see the pair itself."),
            models=dim["models"], diagonal=False,
            settings={"map": {"x": {"property": "separation", "scale": "log"},
                              "y": {"property": f"err/{m_lr}"}, "color": {"property": "class"},
                              "size": {"property": f"abserr/{m_loc}"}}}))
        out.append(dict(id="v-mm", key="dimers", title="Do two models fail on the same structures?", prose=(
            f"{m_lr}'s error against {m_loc}'s, coloured by separation. Points on the diagonal are hard for both — "
            "the reference or the geometry. Points on one axis only are a property of that model. Swap either axis "
            "for any other model in the property list."),
            models=dim["models"], diagonal=True,
            settings={"map": {"x": {"property": f"err/{m_lr}"}, "y": {"property": f"err/{m_loc}"},
                              "color": {"property": "separation", "scale": "log"}}}))
    if "dimers-atoms" in built:
        out.append(dict(id="v-atom", key="dimers-atoms", title="Atoms, not structures", prose=(
            "Every atom of every scored dimer: its DFT Löwdin charge against the formal charge it was assigned, "
            "coloured by fragment. This is the gap between the charge a model is <em>told</em> about and the charge "
            "the electrons actually have — what every charge-equilibration scheme is trying to predict. Each point "
            "is one atomic environment, so selecting it highlights that atom in the structure."),
            models=[], diagonal=True,
            settings={"map": {"x": {"property": "q_lowdin"}, "y": {"property": "q_formal"},
                              "color": {"property": "fragment"}}}))
    for setname, title, prose in (
        ("trimers", "Three-body", "E3 is what is left after every pair contribution is removed, so a model that "
                                  "reproduces dimers and misses this has no cooperative polarisation."),
        ("families_fixedq", "Fixed-charge response", "ΔΔE quartets: composition and total charge are identical in "
                                                     "all four terms, so only the response to a distant charge "
                                                     "survives.")):
        if setname in built:
            b = built[setname]
            m = pick(b["models"], "mace-polar", "so3lr-tauri", "orbmol")
            out.append(dict(id=f"v-{setname}", key=setname, title=title, prose=prose,
                            models=b["models"], diagonal=True,
                            settings={"map": {"x": {"property": f"{b['label']}/REF"},
                                              "y": {"property": f"{b['label']}/{m}"},
                                              "color": {"property": "class"}}}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--data", default=None, help="dataset directory (default: alongside runs, ../lr-sets)")
    ap.add_argument("-o", "--out", default="chemiscope-views.html")
    ap.add_argument("--chemiscope-version", default="1.1.0")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    built = payloads(a.runs, a.data)
    if not built:
        sys.exit("nothing to build: no set had both references and saved predictions")
    summary = [f"{b['n']} {k}" for k, b in built.items() if not k.endswith("-atoms")]

    sections, views = [], []
    for v in view_specs(built):
        views.append({"id": v["id"], "key": v["key"], "settings": v["settings"],
                      "models": v.get("models", []), "diagonal": bool(v.get("diagonal"))})
        sections.append(SECTION.format(id=v["id"], title=v["title"], prose=v["prose"]))

    rank = {"cols": [], "rows": [], "sort": "abs_eps"}
    try:
        from lrbench.diagnostics import collect

        _d = collect(a.runs, with_points=False)
        rank["cols"] = [{"k": "model", "t": "model"}, {"k": "lr", "t": "long range"},
                        {"k": "tail", "t": "tail fraction"}, {"k": "eps", "t": "ε"},
                        {"k": "ddE", "t": "ΔΔE slope"}, {"k": "off8", "t": "8 Å MAE"},
                        {"k": "far12", "t": "≥12 Å MAE"}, {"k": "fmae", "t": "F MAE"},
                        {"k": "level", "t": "level"}]
        for name, r in _d["results"].items():
            S = r["sets"].get("dimers")
            if not S:
                continue
            c = S["classes"].get("ion_pair", {})
            g = lambda tp, n=3: None if not tp or tp[0] is None else round(tp[0], n)  # noqa: E731
            eps = g(c.get("eps"))
            # the fixed-charge quartets live in their own set, not in dimers
            ddE = r["sets"].get("families_fixedq", {}).get("ddE")
            rank["rows"].append({
                "model": r["label"][:36], "ceiling": "oracle" in name,
                "lr": "ceiling" if "oracle" in name else ("yes" if r["has_lr"] else "no"),
                "tail": g(c.get("tail")), "eps": eps, "ddE": g(ddE, 4),
                "off8": g(c.get("off8"), 1), "far12": g(c.get("far12"), 1), "fmae": g(S.get("F_mae"), 1),
                "level": r["level"][:20] + (" ⚠" if r["mismatch"] else ""),
                "abs_eps": abs(eps) if eps is not None else 9e9,
            })
        print(f"[chemiscope] ranking: {len(rank['rows'])} models")
    except Exception as e:
        print(f"[chemiscope] no ranking table ({type(e).__name__}: {e})")

    out = pathlib.Path(a.out).resolve()
    html = HTML.format(
        ranking=RANKING if rank["rows"] else "",
        rank=json.dumps(rank),
        ver=a.chemiscope_version,
        summary=" · ".join(summary) + f" · {len(built.get('dimers', {}).get('models', []))} models · "
               "one embedded dataset per set, reused by every view",
        sections="\n".join(sections),
        data="{" + ",".join(f'"{k}":"{v["blob"]}"' for k, v in built.items()) + "}",
        views=json.dumps(views),
    )
    out.write_text(html)
    print(f"[chemiscope] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, {len(views)} views)")
    if a.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
