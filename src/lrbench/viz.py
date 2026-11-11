"""Chemiscope export of any long-range set (ours or external) in the explorer's property schema
(``tools/explorer/PROPERTIES.md``), so sets and models share maps and colour scales.

Sources: a canonical config (``hf://<repo>/<config>`` or ``<dir>:<config>``), a canonical
``.jsonl``, any ASE-readable file (``.extxyz``...), or an ``https://`` URL to one of those (downloaded into a new empty
temporary directory, read as data only). ``--compare`` merges further sources or chemiscope files (``.json[.gz]``);
every entry carries ``source``. One entry per structure (the explorer's group-entry *families* views are not
produced here). Missing numbers are NaN, missing strings ``"none"``, booleans ``"true"/"false"/"none"``.

Models: ``--model TAG=spec`` (or a bare spec, tag = file stem); ``TAG=spec1,spec2,...`` is a seed set: its mean goes in
``<q>/TAG`` and the seed standard deviation (ddof 1) in ``<q>_sd/TAG``."""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import HF_REPO, DataSet, load_set, needed_terms, structure, term_structure

SCHEMES = ("lowdin", "iao", "mulliken", "esp", "hirshfeld", "mbis")
BOHR = 0.529177210903
R_TAIL = 8.0
# name: (unit, description); target from the name (atom props listed in ATOM)
DESC = {
    "source": ("", "dataset / file the entry comes from"), "entry": ("", "structure"),
    "level": ("", "level of theory of the record's labels (energies are never compared across levels)"),
    "id": ("", "canonical record id"), "set": ("", "canonical config"), "class": ("", "canonical cls"),
    "group": ("", "scan / pair / system key"), "role": ("", "role inside the group"),
    "charge": ("e", "total charge (e)"), "spin": ("", "spin multiplicity 2S+1"), "n_atoms": ("", "number of atoms"),
    "locality": ("", "within_rf / lr_blind / none (PET-OMol-S receptive field 8 A x 3 layers)"),
    "q_fixed_in_pair": ("", "true / false / none: every difference this record enters keeps Q fixed"),
    "flags": ("", "';'-joined canonical flags, none if empty"),
    "partner_copy": ("", "true if the record is a copy from another config"),
    "distance": ("Å", "canonical distance (dimers: d_min; trimers: min inter-fragment distance), Å"),
    "E_int/REF": ("meV", "interaction energy E_AB - E_A - E_B (dimers) / E_ABC - sum E_mono (trimers), meV"),
    "E_mp": ("meV", "multipole-expansion E_int, meV"),
    "tail/REF": ("meV", "offset-free tail dE = E_int(d) - E_int(d_ref), d >= 8 A, d_ref = scan's longest point, meV"),
    "E3/REF": ("meV", "three-body energy (ABC records), meV"),
    "dE3_vs_far/REF": ("meV", "E3(geometry) - E3(far reference), fixed-Q brackets, meV"),
    "gap": ("eV", "HOMO-LUMO gap, eV"), "dipole_norm": ("e·Å", "|dipole| about the centre of mass, e·Å"),
    "q_formal": ("e", "formal unit charge on the unit's first atom, e"),
    "fragment_id": ("", "interaction-energy subsystem label"), "unit_id": ("", "bonded unit"),
    **{f"q_{s}": ("e", f"DFT {s} partial charge, e") for s in SCHEMES},
}
ATOM = {"q_formal", "fragment_id", "unit_id", *(f"q_{s}" for s in SCHEMES)}
MODEL_Q = {"E_int": "meV", "tail": "meV", "E3": "meV", "dE3_vs_far": "meV"}


class Entry:
    def __init__(self, atoms, s: dict, a: dict):
        self.atoms, self.s, self.a = atoms, s, a  # structure props, atom props (name -> (N,) array)


def _num(v):
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _b(v):
    return "none" if v is None else str(bool(v)).lower()


def _arr(v, n):
    return np.full(n, np.nan) if v is None or len(v) != n else np.asarray(v, float)


def _dipole(refs):
    d = refs.get("dipole_au")
    return float(np.linalg.norm(d) * BOHR) if d is not None else np.nan


def _from_record(rec: dict, source: str) -> Entry:
    atoms = structure(rec).atoms()
    refs, n = rec.get("refs") or {}, len(rec["species"])
    flags = rec.get("flags") or []
    s = {"source": source, "entry": "structure", "id": rec["id"], "set": rec.get("set") or "none",
         "level": rec.get("level") or "none",
         "class": rec.get("cls") or "none", "group": rec.get("group") or "none", "role": rec.get("role") or "none",
         "charge": float(rec.get("charge", 0)), "spin": float(rec.get("spin") or 1), "n_atoms": float(n),
         "locality": rec.get("locality") or "none", "q_fixed_in_pair": _b(rec.get("q_fixed_in_pair")),
         "flags": ";".join(flags) or "none", "partner_copy": _b("partner_copy" in flags),
         "distance": _num(rec.get("distance")), "E_mp": 1e3 * _num(refs.get("E_mp")), "gap": _num(refs.get("gap")),
         "dipole_norm": _dipole(refs)}
    a = {"q_formal": _arr(rec.get("formal_charges"), n), "fragment_id": _arr(rec.get("fragment_id"), n),
         "unit_id": _arr(rec.get("unit_id"), n), **{f"q_{x}": _arr(refs.get(f"charges_{x}"), n) for x in SCHEMES}}
    return Entry(atoms, s, a)


def _from_atoms(at, source: str, k: int) -> Entry:
    info, n = at.info, len(at)
    g = lambda *ks: next((info[x] for x in ks if x in info), None)
    fl = g("flags")
    s = {"source": source, "entry": "structure", "id": str(g("id", "name", "key") or f"{source}:{k}"),
         "level": str(g("level") or "none"),
         "set": str(g("set") or "none"), "class": str(g("cls", "class", "config_type") or "none"),
         "group": str(g("group", "pair") or "none"), "role": str(g("role") or "none"),
         "charge": _num(g("charge", "total_charge") or 0), "spin": _num(g("spin_multiplicity", "spin") or 1),
         "n_atoms": float(n), "locality": str(g("locality") or "none"), "q_fixed_in_pair": _b(g("q_fixed_in_pair")),
         "flags": (fl if isinstance(fl, str) else ";".join(fl or [])) or "none",
         "partner_copy": _b(g("partner_copy")), "distance": _num(g("distance", "d_min", "dmin")),
         "E_mp": 1e3 * _num(g("E_mp")), "gap": _num(g("gap")),
         "dipole_norm": _dipole({"dipole_au": g("dipole_au")}),
         "E_int/REF": 1e3 * _num(g("E_int", "eint")), "E3/REF": 1e3 * _num(g("E3"))}
    arr = lambda *ks: next((np.asarray(at.arrays[x], float).reshape(n) for x in ks if x in at.arrays),
                           np.full(n, np.nan))
    a = {"q_formal": arr("formal_charges"), "fragment_id": arr("fragment_id"), "unit_id": arr("unit_id"),
         **{f"q_{x}": arr(f"charges_{x}", f"q_{x}") for x in SCHEMES}}
    return Entry(at, s, a)


def _fetch(url: str) -> str:
    d = tempfile.mkdtemp(prefix="lrbench-viz-")
    p = os.path.join(d, os.path.basename(url.split("?")[0]) or "download")
    urllib.request.urlretrieve(url, p)
    return p


def _dataset(src: str) -> Optional[DataSet]:
    if src.startswith("hf://"):
        repo, _, cfg = src[5:].rpartition("/")
        if repo != HF_REPO:
            raise ValueError(f"hf:// sources: only {HF_REPO}/<config> is known (got {src})")
        return load_set(cfg, "hf")
    if src.endswith(".jsonl"):
        d, f = os.path.split(os.path.abspath(src))
        return load_set(f[:-6], d)
    if ":" in src and not os.path.exists(src) and os.path.isdir(src.rsplit(":", 1)[0]):
        d, cfg = src.rsplit(":", 1)
        return load_set(cfg, d)
    return None


def _anchor_groups(ds: DataSet) -> Dict[str, Dict[str, dict]]:
    """quantity -> record id -> group (E_int, E3, dE3_vs_far) anchored on the group's first positive whole record."""
    out = {"E_int": {}, "E3": {}, "dE3_vs_far": {}}
    for g in ds.groups:
        q = "dE3_vs_far" if g["name"].startswith("dE3_vs_far") else g["kind"]
        main = next((t for t, c in g["terms"] if "#" not in t and c > 0), None)
        if q in out and main is not None:
            out[q].setdefault(main, g)
    return out


def _tail(recs: List[dict], eint: Dict[str, float]) -> Dict[str, float]:
    by = {}
    for r in recs:
        d = r.get("distance")
        if r["id"] in eint and d is not None and d >= R_TAIL - 1e-6 and np.isfinite(eint[r["id"]]):
            by.setdefault(r.get("group") or r["id"], []).append((d, r["id"]))
    out = {}
    for pts in by.values():
        if len(pts) < 2:
            continue
        ref = eint[max(pts)[1]]
        out.update({rid: eint[rid] - ref for _, rid in pts})
    return out


def _entries_ds(ds: DataSet, label: str, models) -> List[Entry]:
    A = _anchor_groups(ds)
    out = [_from_record(r, label) for r in ds.records]
    gv = lambda g, E: sum(c * E[t] for t, c in g["terms"]) if all(t in E for t, _ in g["terms"]) else np.nan
    dft = {q: {rid: g["ref"] for rid, g in A[q].items()} for q in A}
    dft["tail"] = _tail(ds.records, dft["E_int"])
    for r, e in zip(ds.records, out):
        for q in ("E_int", "tail", "E3", "dE3_vs_far"):
            e.s[f"{q}/REF"] = 1e3 * _num(dft[q].get(r["id"]))
    for tag, bes in models:
        terms = needed_terms(ds)
        S = [term_structure(ds, t) for t in terms]
        M = []  # per seed: term -> pred
        for b in bes:
            M.append(dict(zip(terms, b.predict(S))))
        En = [{t: p["energy"] for t, p in m.items() if p.get("energy") is not None} for m in M]
        vals = {}
        for q in ("E_int", "E3", "dE3_vs_far"):
            vals[q] = [{rid: gv(g, E) for rid, g in A[q].items() if not ds.group_flagged(g)} for E in En]
        vals["tail"] = [_tail(ds.records, v) for v in vals["E_int"]]
        for r, e in zip(ds.records, out):
            for q in MODEL_Q:
                v = np.array([_num(x.get(r["id"])) for x in vals[q]]) * 1e3
                ref = e.s[f"{q}/REF"]
                if not np.isfinite(v).all() or not np.isfinite(ref):
                    continue
                e.s[f"{q}/{tag}"] = float(v.mean())
                e.s[f"{q}_err/{tag}"] = float(v.mean() - ref)
                e.s[f"{q}_abserr/{tag}"] = float(abs(v.mean() - ref))
                if len(v) > 1:
                    e.s[f"{q}_sd/{tag}"] = float(v.std(ddof=1))
            n = len(r["species"])
            F = [m.get(r["id"], {}).get("forces") for m in M]
            if r.get("forces") is not None and all(f is not None for f in F):
                e.a[f"F_err/{tag}"] = np.linalg.norm(np.mean(F, 0) - np.asarray(r["forces"], float), axis=1)
            Q = [m.get(r["id"], {}).get("charges") for m in M]
            if all(q is not None and len(q) == n for q in Q):
                e.a[f"q_latent/{tag}"] = np.mean(Q, 0)
    return out


def _models_atoms(out: List[Entry], models):
    from .data import Structure
    S = [Structure(e.s["id"], e.atoms.get_chemical_symbols(), e.atoms.positions, int(e.s["charge"] or 0),
                   int(e.s["spin"] or 1)) for e in out]
    for tag, bes in models:
        M = [b.predict(S) for b in bes]
        for k, e in enumerate(out):
            F = [m[k].get("forces") for m in M]
            ref = e.atoms.arrays.get("forces")
            if ref is None and e.atoms.calc is not None and "forces" in getattr(e.atoms.calc, "results", {}):
                ref = e.atoms.calc.results["forces"]
            if ref is not None and all(f is not None for f in F):
                e.a[f"F_err/{tag}"] = np.linalg.norm(np.mean(F, 0) - np.asarray(ref), axis=1)
            Q = [m[k].get("charges") for m in M]
            if all(q is not None for q in Q):
                e.a[f"q_latent/{tag}"] = np.mean(Q, 0)


def entries(src: str, max_n: int = 0, models=()) -> Tuple[List[Entry], str]:
    label = src
    if src.startswith(("http://", "https://")):
        src = _fetch(src)
    ds = _dataset(src)
    if ds is not None:
        E = _entries_ds(ds, ds.name, models)
        label = ds.name
    else:
        from ase.io import iread
        label = os.path.basename(label)
        E = [_from_atoms(at, label, k) for k, at in enumerate(iread(src, index=":"))]
        if models:
            _models_atoms(E, models)
    return (E[:max_n] if max_n else E), label


def _describe(name: str):
    if name in DESC:
        return DESC[name]
    q, _, tag = name.partition("/")
    base = q.replace("_abserr", "").replace("_err", "").replace("_sd", "")
    unit = {"F_err": "eV/Å", "q_latent": "e"}.get(base, MODEL_Q.get(base, ""))
    what = {"_err": "model - DFT", "_abserr": "|model - DFT|", "_sd": "seed standard deviation (ddof 1)"}
    kind = next((v for k, v in what.items() if q.endswith(k)), "model value (seed mean)")
    return unit, f"{base} {kind}, model {tag}, {unit}".rstrip(", ")


def _props(E: List[Entry]) -> dict:
    names = list(dict.fromkeys([k for e in E for k in e.s] + [k for e in E for k in e.a]))
    out = {}
    for k in names:
        unit, desc = _describe(k)
        if any(k in e.a for e in E):
            vals = np.concatenate([e.a.get(k, np.full(len(e.atoms), np.nan)) for e in E])
            if np.isnan(vals).all():
                continue
            out[k] = {"target": "atom", "values": [float(x) for x in vals], "units": unit, "description": desc}
            continue
        v = [e.s.get(k) for e in E]
        if all(isinstance(x, str) or x is None for x in v):
            vals = [x if x is not None else "none" for x in v]
        else:
            vals = [_num(x) for x in v]
            if np.isnan(vals).all():
                continue
        out[k] = {"target": "structure", "values": vals, "units": unit, "description": desc}
    return out


def categorical_ids(props: dict, limit: int = 50) -> dict:
    """chemiscope interprets a string property as categories only up to `limit` distinct values and warns otherwise:
    such columns (group, id, pair...) become integer `<name>_id` columns (codes in first-seen order)."""
    out = {}
    for k, v in props.items():
        vals = v.get("values") or []
        if vals and all(isinstance(x, str) for x in vals) and len(set(vals)) > limit:
            codes = {}
            out[f"{k}_id"] = dict(v, values=[codes.setdefault(x, len(codes)) for x in vals], units=None,
                                  description=(v.get("description") or k) + " (integer code; string values exceed the "
                                  f"{limit}-category limit)")
            out[f"{k}_id"].pop("units", None)
        else:
            out[k] = v
    return out


def _plain_input(structures, properties, environments, settings, metadata) -> dict:
    """chemiscope 1.x JSON written without the chemiscope package (used when it cannot be imported, e.g. a
    cluster env without IPython); same layout as chemiscope.create_input for ASE structures."""
    S = []
    for a in structures:
        x = a.positions
        st = {"size": len(a), "names": a.get_chemical_symbols(), "x": x[:, 0].tolist(), "y": x[:, 1].tolist(),
              "z": x[:, 2].tolist()}
        if a.pbc.any():
            st["cell"] = a.cell.array.reshape(9).tolist()
        S.append(st)
    d = {"meta": dict(metadata), "structures": S,
         "properties": {k: {kk: v for kk, v in p.items() if v != ""} for k, p in properties.items()},
         "settings": settings}
    if environments is not None:
        d["environments"] = [{"structure": int(i), "center": int(j), "cutoff": float(c)} for i, j, c in environments]
    return d


def build(E: List[Entry], name: str, description: str = "") -> dict:
    try:
        import chemiscope
    except ImportError:
        chemiscope = None
    frames = [e.atoms for e in E]
    props = _props(E)
    env = [(i, j, 3.5) for i, a in enumerate(frames) for j in range(len(a))] if any(
        p["target"] == "atom" for p in props.values()) else None
    err = next((k for k in props if k.startswith("E_int_err/")), None)
    has = lambda k: k in props and any(isinstance(v, (int, float)) and np.isfinite(v) for v in props[k]["values"])
    settings = {"map": {"x": {"property": "distance" if has("distance") else "n_atoms"},
                        "y": {"property": "E_int/REF" if has("E_int/REF") else "charge"},
                        "color": {"property": err or "class"}},
                "structure": [{"atomLabels": False, "unitCell": False, "environments": {"activated": False}}]}
    if env is not None:  # with per-atom environments chemiscope defaults the map to atoms; the axes are per structure
        settings["target"] = "structure"
    props = categorical_ids(props)
    make = chemiscope.create_input if chemiscope is not None else _plain_input
    data = make(
        structures=frames, properties=props, environments=env, settings=settings,
        metadata={"name": name, "description": description or "lrbench viz: one entry per structure; energies meV, "
                  "distances Å, charges e, forces eV/Å; property schema lr-bundle explorer PROPERTIES.md v1",
                  "references": [f"https://huggingface.co/datasets/{HF_REPO}"]})
    return data


def _load_chemiscope(path: str) -> dict:
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def merge(a: dict, b: dict, b_source: str) -> dict:
    """Append chemiscope dataset b to a; properties missing on one side become NaN / "none"; every atom gets an
    environment (cutoff 3.5 A) when any atom property exists."""
    _split_levels(a, b)
    ns, na = len(a["structures"]), sum(s["size"] for s in a["structures"])
    ms, ma = len(b["structures"]), sum(s["size"] for s in b["structures"])
    b["properties"].setdefault("source", {"target": "structure", "values": [b_source] * ms})
    out = dict(a, structures=a["structures"] + b["structures"], properties={})
    for k in dict.fromkeys(list(a["properties"]) + list(b["properties"])):
        pa, pb = a["properties"].get(k), b["properties"].get(k)
        p = dict(pa or pb)
        n1, n2 = (ns, ms) if p["target"] == "structure" else (na, ma)
        fill = "none" if any(isinstance(x, str) for x in p["values"]) else float("nan")
        p["values"] = list(pa["values"] if pa else [fill] * n1) + list(pb["values"] if pb else [fill] * n2)
        out["properties"][k] = p
    if any(p["target"] == "atom" for p in out["properties"].values()):
        env = lambda d: d.get("environments") or [{"structure": i, "center": j, "cutoff": 3.5}
                                                  for i, st in enumerate(d["structures"]) for j in range(st["size"])]
        out["environments"] = env(a) + [dict(e, structure=e["structure"] + ns) for e in env(b)]
    return out


def _levels(d: dict) -> set:
    return set((d["properties"].get("level") or {}).get("values") or []) - {"none"}


def _split_levels(a: dict, b: dict):
    """Different label levels on the two sides: energy columns (meV) get an @<level> suffix per side, so no
    colour scale or column ever mixes levels."""
    la, lb = _levels(a), _levels(b)
    if la == lb or not la or not lb:  # an unlabelled side ("none") is not split
        return
    for d, lv in ((a, la), (b, lb)):
        tag = next(iter(lv)) if len(lv) == 1 else "mixed"
        d["properties"] = {(f"{k}@{tag}" if p.get("units") == "meV" and "@" not in k else k): p
                           for k, p in d["properties"].items()}


def write(data: dict, path: str):
    with (gzip.open if path.endswith(".gz") else open)(path, "wt") as f:
        json.dump(data, f)


def parse_models(specs: Sequence[str], make) -> list:
    """['Fa=a.gguf,b.gguf', 'z.gguf'] -> [(tag, [backend, ...]), ...]"""
    out = []
    for s in specs or []:
        tag, eq, rest = s.partition("=")
        if not eq or tag.startswith(("ase:", "mlipcpp-capi:")) or "/" in tag:
            tag, rest = os.path.splitext(os.path.basename(s.split("?")[0].split(":")[-1]))[0], s
        out.append((tag, [make(x) for x in rest.split(",") if x]))
    return out


def run(source: str, out: str, compare=(), models=(), max_n: int = 0) -> dict:
    E, label = entries(source, max_n, models)
    data = build(E, os.path.basename(out).split(".")[0])
    for c in compare:
        if c.endswith((".json", ".json.gz")):
            other, lab = _load_chemiscope(c), os.path.basename(c)
        else:
            Ec, lab = entries(c, max_n, models)
            other = build(Ec, lab)
        data = merge(data, other, lab)
    write(data, out)
    return data
