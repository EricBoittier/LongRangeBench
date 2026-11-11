#!/usr/bin/env python3
"""One chemiscope dataset across sets and models (plus one file per set) and a self-contained HTML viewer, from the
lrbench viz files and saved predictions of the trained-model run (no new model evaluations).

    python combined_viz.py RUN_DIR OUT_DIR [--chemiscope-version 1.1.1]

Sets: dimers (all classes; in-house + external viz files merged column-wise), trimers, families_fixedq (in-house),
emulation (DFT-pathology points + P2 ORCA dimers_q2, built from results/emulation/*.json), tauri_sapt10k_charged (seed-0
sample of 150 charged SAPT10k dimers, separate-monomer E_int from the Table II predictions; SAPT2+(3)(CCD) reference).
Per structure: set, class, distance, split, seen_by, verdict (emulation flags), E_int/REF and, per model,
E_int/M, E_int_err/M (model - REF), E_int_abserr/M, and for models with charges q_A/M, dq_A/M (vs DFT Loewdin fragment sum).
Per atom: q_lowdin, q_frag/REF (DFT fragment sum on each atom), fragment_id, q_latent/M where available.
Map: x = distance (log), y = E_int_err/mace-polar, colour = class. String properties are kept under chemiscope's
21-category limit (seen_by and verdict are collapsed to a few labels)."""
import base64
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")

KEEP_INHOUSE = ["Z", "Fa", "Fb", "FQ", "frag2B", "ctl2B", "g100"]
EXTERNAL = ["mace-polar", "mace-polar-l", "mace-omol", "mace-h", "so3lr", "so3lr-tauri-s", "so3lr-tauri", "so3lr-tauri-l",
            "maceles-off", "caceles-water"]
NAN = float("nan")


def load(p):
    return json.load(gzip.open(p) if p.endswith(".gz") else open(p))


def seen_label(v):
    v = str(v)
    if v in ("unseen", "none", ""):
        return "unseen"
    parts = [x for x in v.split(",") if x]
    store = any(not x.startswith("train_aug") and not x.endswith(":monomer") for x in parts)
    mono = any(x.endswith(":monomer") for x in parts)
    aug = "train_aug" in parts
    return "+".join([x for x, f in (("store", store), ("store:monomer", mono and not store), ("train_aug", aug)) if f]) or "unseen"


def model_cols(P, models, nstruct):
    out = {}
    for m in models:
        for k in ("E_int", "E_int_err", "E_int_abserr", "E_int_sd"):
            if f"{k}/{m}" in P:
                out[f"{k}/{m}"] = P[f"{k}/{m}"]
    return out


def frag_charges(d, models):
    """q_A/M, dq_A/M (structure) and q_frag/REF (atom) from q_latent/M, q_lowdin, fragment_id."""
    P = d["properties"]; sizes = [s["size"] for s in d["structures"]]; off = np.concatenate([[0], np.cumsum(sizes)])
    if "fragment_id" not in P or "q_lowdin" not in P:
        return {}
    fid = np.asarray(P["fragment_id"]["values"], float); ql = np.asarray(P["q_lowdin"]["values"], float)
    qfrag = np.full(len(fid), NAN); out = {}
    qa_ref = np.full(len(sizes), NAN)
    for i in range(len(sizes)):
        sl = slice(off[i], off[i + 1]); f = fid[sl]
        if np.all(np.isfinite(f)) and np.all(np.isfinite(ql[sl])):
            for k in set(f.tolist()):
                m = f == k; qfrag[sl][m] = ql[sl][m].sum()
            qa_ref[i] = ql[sl][f == 0].sum()
    out["q_frag/REF"] = dict(target="atom", values=qfrag.tolist(), units="e",
                             description="DFT Loewdin fragment charge, shown on each atom of the fragment")
    out["q_A/REF"] = dict(target="structure", values=qa_ref.tolist(), units="e", description="DFT Loewdin charge of fragment A")
    for m in models:
        if f"q_latent/{m}" not in P:
            continue
        q = np.asarray(P[f"q_latent/{m}"]["values"], float); qa = np.full(len(sizes), NAN)
        for i in range(len(sizes)):
            sl = slice(off[i], off[i + 1]); f = fid[sl]
            if np.all(np.isfinite(f)):
                qa[i] = q[sl][f == 0].sum()
        out[f"q_A/{m}"] = dict(target="structure", values=qa.tolist(), units="e", description=f"{m}: fragment-A charge (sum of model charges)")
        out[f"dq_A/{m}"] = dict(target="structure", values=(qa - qa_ref).tolist(), units="e",
                                description=f"{m}: fragment-A charge minus DFT Loewdin (emulation charge check)")
    return out


def base_props(d, set_name):
    P = d["properties"]; n = len(d["structures"])
    g = lambda k, dflt: P[k]["values"] if k in P else [dflt] * n
    out = {"set": dict(target="structure", values=[set_name] * n),
           "class": dict(target="structure", values=[str(x) for x in g("class", "none")]),
           "distance": dict(target="structure", values=[NAN if x is None else float(x) for x in g("distance", NAN)], units="Å"),
           "split": dict(target="structure", values=[str(x) for x in g("split", "none")]),
           "seen_by": dict(target="structure", values=[seen_label(x) for x in g("seen_by", "unseen")]),
           "id": dict(target="structure", values=[str(x) for x in g("id", "none")]),
           "E_int/REF": dict(target="structure", values=g("E_int/REF", NAN), units="meV")}
    for k, u in (("gap", "Å"), ("n_atoms", None), ("charge", "e")):
        if k in P:
            out[k] = dict(target="structure", values=[NAN if x is None else float(x) for x in P[k]["values"]], units=u)
    if "role" in P and len(set(map(str, P["role"]["values"]))) <= 21:
        out["role"] = dict(target="structure", values=[str(x) for x in P["role"]["values"]])
    for k in ("q_lowdin", "fragment_id"):
        if k in P:
            out[k] = dict(target="atom", values=P[k]["values"], units="e" if k == "q_lowdin" else None)
    return out


def from_viz(run, set_name, models):
    base = load(f"{run}/viz/{set_name}_inhouse.json.gz")
    props = base_props(base, set_name)
    P = base["properties"]
    for m in KEEP_INHOUSE:
        props.update({k: dict(v, target="structure") for k, v in model_cols(P, [m], 0).items()})
        if f"q_latent/{m}" in P:
            props[f"q_latent/{m}"] = dict(P[f"q_latent/{m}"], target="atom")
    ids = P["id"]["values"]
    for m in EXTERNAL:
        f = f"{run}/viz/{set_name}_{m}.json.gz"
        if not os.path.exists(f):
            continue
        e = load(f); Q = e["properties"]
        if Q["id"]["values"] != ids:
            print("skip (order differs)", f); continue
        props.update({k: dict(v, target="structure") for k, v in model_cols(Q, [m], 0).items()})
        if f"q_latent/{m}" in Q:
            props[f"q_latent/{m}"] = dict(Q[f"q_latent/{m}"], target="atom")
    tmp = dict(base, properties={**P, **props})
    props.update(frag_charges(tmp, KEEP_INHOUSE + EXTERNAL))
    return dict(structures=base["structures"], properties=props)


def from_emulation(run):
    R = [json.loads(l) for l in open(f"{run}/emulation/data/emulation.jsonl")]
    G = {g["terms"][0][0]: g for g in json.load(open(f"{run}/emulation/data/emulation.groups.json"))}
    V = {p["id"]: p for p in json.load(open(f"{run}/results/emulation_flags.json"))["points"]}
    S, n = [], len(R)
    for r in R:
        x = np.asarray(r["positions"])
        S.append(dict(size=len(r["species"]), names=r["species"], x=x[:, 0].tolist(), y=x[:, 1].tolist(), z=x[:, 2].tolist()))
    vlab = {"a": "bad label (SCF not lowest)", "b": "DFT pathology (emulation only)", "clean": "clean"}
    props = {"set": dict(target="structure", values=["emulation" if "full" in r["id"] else "dimers_q2" for r in R]),
             "class": dict(target="structure", values=[r["cls"] for r in R]),
             "distance": dict(target="structure", values=[r["distance"] for r in R], units="Å"),
             "split": dict(target="structure", values=["heldout"] * n), "seen_by": dict(target="structure", values=["unseen"] * n),
             "id": dict(target="structure", values=[r["id"] for r in R]),
             "verdict": dict(target="structure", values=[vlab[V[r["id"]]["verdict"]] for r in R]),
             "E_int/REF": dict(target="structure", values=[G[r["id"]]["ref"] * 1e3 for r in R], units="meV"),
             "q_lowdin": dict(target="atom", values=[q for r in R for q in (r["refs"]["charges_lowdin"] or [NAN] * len(r["species"]))], units="e"),
             "fragment_id": dict(target="atom", values=[k for r in R for k in r["fragment_id"]])}
    names = {"pet-omol-m": "PET-OMol-M", "pet-omol-l": "PET-OMol-L", "Fa-s13": "Fa", "Fb-s13": "Fb", "FQ-s13": "FQ"}
    for p in sorted(glob.glob(f"{run}/results/emulation/*.json")):
        X = json.load(open(p)); m = names.get(X.get("label_short") or X["label"], X.get("label_short") or X["label"])
        blk = X["sets"]["emulation"]; P = blk["predictions"]; qf = blk.get("qfrag") or {}
        e = []
        for r in R:
            try:
                e.append(sum(c * np.nanmean([v for v in P[t] if v is not None]) for t, c in G[r["id"]]["terms"]) * 1e3)
            except (KeyError, ZeroDivisionError, TypeError):
                e.append(NAN)
        e = np.asarray(e, float); ref = np.asarray(props["E_int/REF"]["values"], float)
        props[f"E_int/{m}"] = dict(target="structure", values=e.tolist(), units="meV")
        props[f"E_int_err/{m}"] = dict(target="structure", values=(e - ref).tolist(), units="meV")
        props[f"E_int_abserr/{m}"] = dict(target="structure", values=np.abs(e - ref).tolist(), units="meV")
        if qf:
            qa = [np.nanmean([v[0] for v in qf[r["id"]] if v]) if r["id"] in qf else NAN for r in R]
            props[f"q_A/{m}"] = dict(target="structure", values=qa, units="e")
            props[f"dq_A/{m}"] = dict(target="structure", values=[a - (r["refs"]["qA_lowdin"] if r["refs"]["qA_lowdin"] is not None else NAN)
                                                                   for a, r in zip(qa, R)], units="e")
    ds = dict(structures=S, properties=props)
    props.update({k: v for k, v in frag_charges(ds, []).items()})
    return ds


def from_tauri(run, n=150):
    from ase.io import read
    A = read(f"{run}/scans/tauri_table2.extxyz", ":")
    orig = [i for i in range(0, len(A), 4) if A[i].info["set_name"] == "sapt10k" and A[i].info["charged"]]
    keep = sorted(np.random.default_rng(0).choice(len(orig), min(n, len(orig)), replace=False))
    idx = [orig[k] for k in keep]
    S = [dict(size=len(A[i]), names=A[i].get_chemical_symbols(), x=A[i].positions[:, 0].tolist(),
              y=A[i].positions[:, 1].tolist(), z=A[i].positions[:, 2].tolist()) for i in idx]
    ref = np.array([A[i].info["ref_Eint_meV"] for i in idx])
    dmin = [float(np.min(np.linalg.norm(A[i + 2].positions[:, None] - A[i + 3].positions[None], axis=-1))) for i in idx]
    nat = sum(len(A[i]) for i in idx)
    props = {"set": dict(target="structure", values=["tauri_sapt10k_charged"] * len(idx)),
             "class": dict(target="structure", values=["sapt10k_charged"] * len(idx)),
             "distance": dict(target="structure", values=dmin, units="Å",
                              description="closest-atom distance between the two monomers (bound geometry)"),
             "split": dict(target="structure", values=["none"] * len(idx)), "seen_by": dict(target="structure", values=["unseen"] * len(idx)),
             "id": dict(target="structure", values=[A[i].info["system_id"] for i in idx]),
             "E_int/REF": dict(target="structure", values=ref.tolist(), units="meV",
                               description="SAPT2+(3)(CCD)/aug-cc-pVTZ (not wB97M-V)"),
             "q_lowdin": dict(target="atom", values=[NAN] * nat, units="e"), "fragment_id": dict(target="atom", values=[NAN] * nat)}
    names = {"Z": "Z", "Fa-s13": "Fa", "Fb-s13": "Fb", "FQ-s13": "FQ", "ctl2B": "ctl2B", "pet-omol-m": "PET-OMol-M", "pet-omol-l": "PET-OMol-L"}
    for p in sorted(glob.glob(f"{run}/pred_c27/*/tauri_table2.json") + glob.glob(f"{run}/pred_external/*/tauri_table2.json")):
        lab = os.path.basename(os.path.dirname(p)); m = names.get(lab, lab)
        fr = json.load(open(p))["frames"]
        E = lambda j: NAN if fr[j].get("energy") is None else fr[j]["energy"]
        e = np.array([(E(i) - E(i + 2) - E(i + 3)) * 1e3 for i in idx])
        props[f"E_int/{m}"] = dict(target="structure", values=e.tolist(), units="meV")
        props[f"E_int_err/{m}"] = dict(target="structure", values=(e - ref).tolist(), units="meV")
        props[f"E_int_abserr/{m}"] = dict(target="structure", values=np.abs(e - ref).tolist(), units="meV")
    return dict(structures=S, properties=props)


def concat(dsets):
    S, P = [], {}
    names = sorted({k for d in dsets for k in d["properties"]})
    for k in names:
        tgt = next(d["properties"][k]["target"] for d in dsets if k in d["properties"])
        vals = []
        for d in dsets:
            n = len(d["structures"]) if tgt == "structure" else sum(s["size"] for s in d["structures"])
            if k in d["properties"]:
                vals += list(d["properties"][k]["values"])
            else:
                isstr = any(isinstance(v, str) for dd in dsets if k in dd["properties"] for v in dd["properties"][k]["values"][:5])
                vals += ["none" if isstr else NAN] * n
        meta = next(d["properties"][k] for d in dsets if k in d["properties"])
        P[k] = {kk: vv for kk, vv in meta.items() if kk not in ("values",)} | {"values": vals}
    for d in dsets:
        S += d["structures"]
    return S, P


def clean(v):
    return None if isinstance(v, float) and not np.isfinite(v) else v


def write(path, S, P, name, desc):
    env = [dict(structure=i, center=j, cutoff=3.5) for i, s in enumerate(S) for j in range(s["size"])]
    from lrbench.viz import categorical_ids
    P = {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in P.items()
         if any(isinstance(x, str) or (x is not None and np.isfinite(x)) for x in v["values"])}  # drop all-NaN columns
    P = categorical_ids(P)  # id, group... -> integer *_id columns (chemiscope's 50-category limit)
    for k, v in P.items():
        if v.get("units") is None:
            v.pop("units", None)
    num = [k for k in P if P[k]["target"] == "structure" and isinstance(next((x for x in P[k]["values"] if x is not None), ""), (int, float))]
    pick = lambda *ks: next((k for k in ks if k in num), num[0])
    xk = pick("distance", "gap", "E_int/REF", "n_atoms")
    yk = pick("E_int_err/mace-polar", *[k for k in num if k.startswith("E_int_err/")], "E_int/REF", "n_atoms", "charge")
    settings = {"target": "structure", "map": {"x": {"property": xk, **({"scale": "log"} if xk == "distance" else {})},
                                               "y": {"property": yk}, "color": {"property": "class" if "class" in P else num[0]}},
                "structure": [{"atomLabels": False, "unitCell": False, "environments": {"activated": False}}]}
    d = dict(meta=dict(name=name, description=desc, authors=["lrbench trained-model run"]), structures=S, properties=P,
             environments=env, settings=settings)
    with gzip.open(path, "wt") as f:  # NaN tokens, as chemiscope.write_input writes them (chemiscope.org reads them)
        f.write(json.dumps(d))
    strict = dict(d, properties={k: dict(v, values=[clean(x) for x in v["values"]]) for k, v in P.items()})
    with gzip.open(path.replace(".json.gz", ".strict.json.gz"), "wt") as f:  # null for NaN: for JSON.parse in the HTML page
        f.write(json.dumps(strict, allow_nan=False))
    cats = {k: len(set(v["values"])) for k, v in P.items() if v["values"] and isinstance(v["values"][0], str)}
    print(path, len(S), "structures", len(P), "properties", f"{os.path.getsize(path) / 1e6:.2f} MB gz", {k: n for k, n in cats.items() if n > 50})
    return d


HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>lrbench combined viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chemiscope@{ver}/dist/chemiscope.min.js"></script>
<style>:root{{--bg:#fff;--fg:#1a1a1a;--mut:#555}}@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16181c;--fg:#e8e8e8;--mut:#aaa}}}}
:root[data-theme="dark"]{{--bg:#16181c;--fg:#e8e8e8;--mut:#aaa}}
body{{background:var(--bg);color:var(--fg);font-family:Arial,Helvetica,sans-serif;font-size:13px;margin:16px}}
#bar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px}} p{{color:var(--mut);max-width:1100px}}
.visualizer-container{{display:flex;flex-wrap:wrap;gap:8px}} .visualizer-column,.visualizer-column-right{{flex:1 1 420px;min-width:300px}}
#cs-map{{height:520px}} #cs-structure{{height:420px}}</style></head><body>
<div id="bar"><b>lrbench: sets × models</b><label>dataset <select id="sel"></select></label></div>
<p>Map: x = d_min (log), y = signed E_int error of the selected model (default MACE-POLAR-1 M), colour = class; switch axes
to compare two models' errors (E_int_err/A vs E_int_err/B) or a charge deviation (dq_A/M) against an error. Atom colours:
choose q_frag/REF (DFT Löwdin fragment charge) or q_latent/M in the structure panel. References are ωB97M-V/def2-TZVPD
except the SAPT10k sample (SAPT2+(3)(CCD)). Oracles use fragment information the models do not get.</p>
<div id="host">loading…</div>
<script>
const DATA = {{{data}}};
async function inflate(b64) {{
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(s).text());
}}
let viewer = null;
async function show(name) {{
  if (viewer) {{ viewer.remove(); viewer = null; }}
  const host = document.getElementById("host");
  host.innerHTML = `<div class="visualizer-container"><div class="visualizer-column-right"><div id="cs-meta"></div><div id="cs-map"></div></div>
    <div class="visualizer-column"><div id="cs-structure"></div><div id="cs-info"></div></div></div>`;
  try {{
    const ds = await inflate(DATA[name]);
    for (const p of Object.values(ds.properties)) {{  // null stands for NaN in the embedded strict JSON
      if (p.values.some(v => typeof v === "number")) p.values = p.values.map(v => v === null ? NaN : v);
    }}
    viewer = await Chemiscope.DefaultVisualizer.load({{map: "cs-map", info: "cs-info", meta: "cs-meta", structure: "cs-structure"}}, ds);
    window.viewer = viewer;
  }} catch (e) {{ host.innerText = "chemiscope error: " + e; console.error(e); }}
}}
const sel = document.getElementById("sel");
for (const k of Object.keys(DATA)) {{ const o = document.createElement("option"); o.text = k; sel.add(o); }}
if (location.hash && DATA[location.hash.slice(1)]) sel.value = location.hash.slice(1);  // #<dataset> selects one
sel.onchange = () => show(sel.value);
show(sel.value);
</script></body></html>
"""


def main(run, out, *a):
    ver = a[a.index("--chemiscope-version") + 1] if "--chemiscope-version" in a else "1.1.1"
    os.makedirs(out, exist_ok=True)
    sets = {"dimers": from_viz(run, "dimers", None), "trimers": from_viz(run, "trimers", None),
            "families_fixedq": from_viz(run, "families_fixedq", None), "emulation": from_emulation(run),
            "tauri_sapt10k_charged": from_tauri(run)}
    files = {}
    for k, d in sets.items():
        S, P = concat([d])
        files[k] = f"{out}/{k}.json.gz"
        write(files[k], S, P, f"lrbench {k}", f"lrbench trained-model run: {k}")
    S, P = concat(list(sets.values()))
    files["combined"] = f"{out}/combined.json.gz"
    write(files["combined"], S, P, "lrbench combined", "lrbench trained-model run: dimers, trimers, families_fixedq, emulation + dimers_q2, SAPT10k charged sample")
    data = ",".join(f'"{k}": "{base64.b64encode(open(f.replace(".json.gz", ".strict.json.gz"), "rb").read()).decode()}"' for k, f in
                    [("combined", files["combined"])] + [(k, files[k]) for k in sets])
    html = HTML.format(ver=ver, data=data)
    open(f"{out}/combined_viewer.html", "w").write(html)
    print(f"{out}/combined_viewer.html", f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
