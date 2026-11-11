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
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap"
      rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chemiscope@{ver}/dist/chemiscope.min.js"></script>
<style>
:root {{ --bg:#fcfcfb; --fg:#1a1a1a; --mut:#767676; --rule:#e2e2e2; --blue:#0072B2; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#15171a; --fg:#e9e9e7; --mut:#a6a6a3; --rule:#2c2f34; }} }}
:root[data-theme="dark"] {{ --bg:#15171a; --fg:#e9e9e7; --mut:#a6a6a3; --rule:#2c2f34; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); margin:0;
  font-family:"Crimson Pro",Crimson,Charter,Georgia,serif; font-size:17px; line-height:1.55; }}
header, section {{ max-width:1180px; margin:0 auto; padding:0 24px; }}
header {{ padding-top:44px; padding-bottom:8px; }}
h1 {{ font-size:34px; font-weight:600; margin:0 0 6px; letter-spacing:-0.01em; }}
h2 {{ font-size:23px; font-weight:600; margin:46px 0 4px; }}
.lede, p {{ color:var(--fg); max-width:72ch; }}
.sub, .meta {{ color:var(--mut); font-size:15px; max-width:76ch; }}
hr {{ border:0; border-top:1px solid var(--rule); margin:34px 0 0; }}
.view {{ margin:14px 0 30px; }}
.panes {{ display:flex; flex-wrap:wrap; gap:10px; }}
.left {{ flex:1 1 560px; min-width:320px; }} .right {{ flex:1 1 420px; min-width:300px; }}
.cs-map {{ height:470px; }} .cs-structure {{ height:380px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:14px;
  background:color-mix(in srgb, var(--rule) 55%, transparent); padding:1px 4px; border-radius:3px; }}
.err {{ color:#D55E00; }}
footer {{ max-width:1180px; margin:40px auto 60px; padding:0 24px; color:var(--mut); font-size:15px; }}
@media (max-width:700px) {{ body {{ font-size:16px; }} header, section, footer {{ padding:0 16px; }}
  .cs-map {{ height:380px; }} }}
</style></head><body>
<header>
<h1>Long-range explorer</h1>
<p class="lede">Every scored structure, with its reference value and each model's answer on it. The views below
start pointed at a particular question, and then they are yours: chemiscope's own controls change the axes, the
colour, the point size and the atom colouring, so any property here can be plotted against any other.</p>
<p class="meta">{summary}</p>
</header>
{sections}
<footer>
<p>Built by <code>tools/build_chemiscope_views.py</code> from the canonical records and the per-term energies saved
by <code>lrbench eval --save-predictions</code>. A model's value and its reference are the same linear combination
of predicted terms, so a point compares like with like. Records flagged as DFT pathologies are excluded. Energies
in meV; a model trained at another level of theory is marked in the ranking table, and its energies are not
comparable with these references.</p>
</footer>
<script>
const DATA = {data};
const VIEWS = {views};
async function inflate(b64) {{
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(s).text());
}}
const cache = {{}};
async function dataset(key) {{
  if (!cache[key]) cache[key] = await inflate(DATA[key]);
  // null stands for NaN in strict JSON; chemiscope wants the number
  const ds = structuredClone(cache[key]);
  for (const p of Object.values(ds.properties)) {{
    if (p.values.every(v => v === null || typeof v === "number"))
      p.values = p.values.map(v => v === null ? NaN : v);
  }}
  return ds;
}}
(async () => {{
  for (const v of VIEWS) {{
    const host = document.getElementById(v.id);
    try {{
      const ds = await dataset(v.key);
      if (v.settings) ds.settings = v.settings;
      await Chemiscope.DefaultVisualizer.load(
        {{ map: v.id + "-map", info: v.id + "-info", meta: v.id + "-meta", structure: v.id + "-structure" }}, ds);
    }} catch (e) {{
      host.innerHTML = '<p class="err">chemiscope could not load this view: ' + e + '</p>';
      console.error(e);
    }}
  }}
}})();
</script></body></html>
"""

SECTION = """<section>
<h2>{title}</h2>
<p>{prose}</p>
<div class="view" id="{id}">
  <div class="panes">
    <div class="left"><div id="{id}-meta"></div><div class="cs-map" id="{id}-map"></div></div>
    <div class="right"><div class="cs-structure" id="{id}-structure"></div><div id="{id}-info"></div></div>
  </div>
</div>
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
            settings={"map": {"x": {"property": "separation", "scale": "log"},
                              "y": {"property": f"err/{m_lr}"}, "color": {"property": "class"},
                              "size": {"property": f"abserr/{m_loc}"}}}))
        out.append(dict(id="v-mm", key="dimers", title="Do two models fail on the same structures?", prose=(
            f"{m_lr}'s error against {m_loc}'s, coloured by separation. Points on the diagonal are hard for both — "
            "the reference or the geometry. Points on one axis only are a property of that model. Swap either axis "
            "for any other model in the property list."),
            settings={"map": {"x": {"property": f"err/{m_lr}"}, "y": {"property": f"err/{m_loc}"},
                              "color": {"property": "separation", "scale": "log"}}}))
    if "dimers-atoms" in built:
        out.append(dict(id="v-atom", key="dimers-atoms", title="Atoms, not structures", prose=(
            "Every atom of every scored dimer: its DFT Löwdin charge against the formal charge it was assigned, "
            "coloured by fragment. This is the gap between the charge a model is <em>told</em> about and the charge "
            "the electrons actually have — what every charge-equilibration scheme is trying to predict. Each point "
            "is one atomic environment, so selecting it highlights that atom in the structure."),
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
        views.append({"id": v["id"], "key": v["key"], "settings": v["settings"]})
        sections.append(SECTION.format(id=v["id"], title=v["title"], prose=v["prose"]))

    out = pathlib.Path(a.out).resolve()
    html = HTML.format(
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
