"""Per-point material for the notebook's multivalent-ion and charge-transfer / leakage panels.

The aggregate result rows say *how wrong* a model is per class; these two panels need the points themselves: every
interaction energy of a charged pair, with its separation, the fragments' formal charges, the reference fragment
charges (DFT Löwdin sums, where the set has them), the model's fragment charges (``qfrag``, from
``lrbench eval --save-predictions``, where the model returns charges), and every charge-transfer and pathology label
the data carries -- kept, never dropped.

    from lrbench.ionct import points
    pts = points(result_json, "dimers", data="lr-sets")      # {"cols": COLS, "rows": [[...], ...]}

Rows are compact lists (the notebook embeds them for WebAssembly), one per ``E_int`` group whose AB record has two
fragments of which at least one is charged, kept when the pair is multivalent (|q| >= 2), CT-labelled, a like-charge
pair, or has reference fragment charges. A model value is the same linear combination of saved term energies that
defines the reference, members averaged, exactly as :func:`lrbench.diagnostics.parity` forms it; a group with any
failed term has model ``None`` -- never zero.

Data are found **only** in the directory passed as ``data`` (``<data>/<set>.jsonl`` or ``<data>/canonical/...``),
or relative to ``base`` when the result recorded a relative path. The absolute path a result recorded is never
followed, and anything that resolves under ``private`` is refused: a result scored on an internal-only set and left in
``runs/`` must not be frozen into a published notebook. Internal sets are read by :func:`private_points`, which the
notebook calls live only.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional

KE = 14.399645  # e^2/(4 pi eps0), eV Å

#: column order of a row; the notebook indexes by name through this list
COLS = ["set", "id", "pair", "cls", "d", "r", "qA", "qB", "spin", "ref", "model", "lwA", "lwB", "qmA", "qmB",
        "ct", "path", "phys", "level", "geo"]

#: emulation-score tags that the physics score excludes (lrbench.emulation.PHYSICS_EXCLUDE_TAGS), matched as substrings
#: so both ``ct_fractional`` and ``dft:pathology:ct_fractional`` count
PHYSICS_EXCLUDE = ("ct_fractional", "ct_ground_state", "unbound_anion", "spin_contaminated")
NEVER_SCORED = "dft_pathology_do_not_score"
_CT = re.compile(r"(^|[:_])ct([:_]|$)|frac_charge")
#: sets whose rows are 2-D maps or training data, not separations of a fixed pair: left out of these panels
SKIP_SETS = {"maps2d", "pulled_apart", "toy"}


def ct_labels(flags: Iterable[str], cls: Optional[str]) -> List[str]:
    """Every charge-transfer label on a point: record flags, group flags and a CT class (``ion_pair_ct_ambiguous``,
    ``ct:ion_polar``), de-duplicated in order."""
    out = [f for f in flags if _CT.search(str(f))]
    if cls and _CT.search(str(cls)):
        out.append(f"class:{cls}")
    return list(dict.fromkeys(out))


def pathology(flags: Iterable[str]) -> int:
    """0 clean, 1 DFT-pathology-tagged (emulation target, excluded from the physics score), 2 never scored."""
    flags = [str(f) for f in flags]
    if NEVER_SCORED in flags:
        return 2
    if any(f.startswith("dft:pathology") or any(t in f for t in PHYSICS_EXCLUDE) for f in flags):
        return 1
    return 0


def _read_set(name: str, data: Optional[str], base: Optional[str], recorded: Optional[str],
              private: Optional[str], cache: dict):
    """(records by id, groups) of a set, from ``data`` (or base-relative), or None. Never an absolute recorded path,
    never anything under ``private``."""
    cands = []
    for root in [data] if data else []:
        cands += [os.path.join(root, f"{name}.jsonl"), os.path.join(root, "canonical", f"{name}.jsonl")]
    if recorded and not os.path.isabs(recorded) and base:
        cands.append(os.path.join(base, recorded))
    priv = os.path.realpath(private) + os.sep if private else None
    for p in cands:
        if not os.path.exists(p):
            continue
        rp = os.path.realpath(p)
        if priv and (rp + os.sep).startswith(priv):
            continue
        if rp in cache:
            return cache[rp]
        gp = p[: -len(".jsonl")] + ".groups.json"
        if not os.path.exists(gp):
            continue
        with open(p) as f:
            recs = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        with open(gp) as f:
            groups = json.load(f)
        cache[rp] = (recs, groups)
        return cache[rp]
    return None


def _geo(r: dict) -> str:
    """Short geometry key (species, positions to 1 mÅ, spin): the same structure listed in two sets is one point."""
    h = hashlib.sha1()
    h.update(json.dumps([r.get("species"), [[round(float(x), 3) for x in p] for p in r.get("positions") or []],
                         r.get("spin")]).encode())
    return h.hexdigest()[:10]


def _frag_sums(q, fid, n=2):
    if q is None or fid is None or len(q) != len(fid):
        return None
    s = [0.0] * n
    for qi, k in zip(q, fid):
        if 0 <= int(k) < n:
            s[int(k)] += float(qi)
    return s


def _ref_fragment_charges(r: dict):
    """DFT Löwdin fragment sums [A, B]: ``refs.fragment_charges_lowdin``, else summed ``charges_lowdin``, else
    ``qA_lowdin`` with B = Q - A."""
    refs = r.get("refs") or {}
    fq = refs.get("fragment_charges_lowdin")
    if fq is not None and len(fq) == 2:
        return [float(fq[0]), float(fq[1])]
    s = _frag_sums(refs.get("charges_lowdin"), r.get("fragment_id"))
    if s is not None:
        return s
    if refs.get("qA_lowdin") is not None:
        a = float(refs["qA_lowdin"])
        return [a, float(r.get("charge", 0)) - a]
    return None


def _centroid_distance(r: dict) -> Optional[float]:
    ex = r.get("extra") or {}
    if isinstance(ex.get("r_com"), (int, float)):
        return float(ex["r_com"])
    fid, pos = r.get("fragment_id"), r.get("positions")
    if fid is None or pos is None:
        return None
    c = []
    for k in (0, 1):
        p = [x for x, f in zip(pos, fid) if int(f) == k]
        if not p:
            return None
        c.append([sum(v[i] for v in p) / len(p) for i in range(3)])
    return math.dist(c[0], c[1])


def _pair_name(r: dict, g: dict) -> str:
    m, ex = g.get("meta") or {}, r.get("extra") or {}
    p = ex.get("pair") or m.get("pair") or m.get("system") or m.get("scan") or r.get("group") or g.get("name", "?")
    p = str(p)
    if r.get("spin", 1) == 3 and not p.endswith("_t"):
        p += "_t"
    return p


def _r4(x, n=4):
    return None if x is None else round(float(x), n)


def keep(qA: int, qB: int, ct: list, lw) -> bool:
    """Which charged pairs these panels carry: multivalent, CT-labelled, like-charge, or with reference charges."""
    if qA == 0 and qB == 0:
        return False
    return max(abs(qA), abs(qB)) >= 2 or bool(ct) or qA * qB > 0 or lw is not None


def rows_for_set(recs: dict, groups: list, setname: str, P: Optional[dict] = None,
                 qfrag: Optional[dict] = None) -> list:
    """Compact rows (:data:`COLS`) for one set; ``P`` the saved term energies, ``qfrag`` the saved fragment charges."""
    out = []
    for g in groups or []:
        if g.get("kind") != "E_int" or g.get("ref") is None or not g.get("terms"):
            continue
        rid = str(g["terms"][0][0]).split("#", 1)[0]
        r = recs.get(rid)
        if r is None:
            continue
        fq = r.get("fragment_charges") or []
        if len(fq) != 2:
            continue
        qA, qB = int(round(fq[0])), int(round(fq[1]))
        m = g.get("meta") or {}
        flags = list(dict.fromkeys([*(r.get("flags") or []), *(m.get("flags") or [])]))
        ct = ct_labels(flags, r.get("cls") or m.get("cls"))
        lw = _ref_fragment_charges(r)
        if not keep(qA, qB, ct, lw):
            continue
        path = pathology(flags)
        phys = path == 0 and not ct
        d = next((m[k] for k in ("dmin", "distance", "d_min", "d") if isinstance(m.get(k), (int, float))),
                 r.get("distance"))
        model = None
        if P:
            v, ok = 0.0, True
            for term, coef in g["terms"]:
                vals = [x for x in (P.get(term) or []) if isinstance(x, (int, float))]
                if not vals:
                    ok = False
                    break
                v += float(coef) * (sum(vals) / len(vals))
            model = round(1e3 * v, 3) if ok else None
        qm = None
        if qfrag and rid in qfrag:
            per = [p for p in qfrag[rid] if p is not None and len(p) == 2]
            if per:
                qm = [sum(p[k] for p in per) / len(per) for k in (0, 1)]
        out.append([setname, rid, _pair_name(r, g), str(r.get("cls") or m.get("cls") or "?"), _r4(d, 3),
                    _r4(_centroid_distance(r), 3), qA, qB, int(r.get("spin") or 1), round(1e3 * float(g["ref"]), 3),
                    model, _r4(lw and lw[0]), _r4(lw and lw[1]), _r4(qm and qm[0]), _r4(qm and qm[1]), ct, path,
                    phys, str(r.get("level") or m.get("theory") or "?"), _geo(r)])
    return out


def points(J: dict, setname: str, data: Optional[str] = None, base: Optional[str] = None,
           private: Optional[str] = None, cache: Optional[dict] = None) -> Optional[dict]:
    """``{"cols", "rows"}`` for one result and set, or None when the set's data are not found where allowed."""
    if setname in SKIP_SETS or setname.startswith("toy"):
        return None
    S = (J.get("sets") or {}).get(setname) or {}
    rec = ((J.get("data") or {}).get(setname) or {})
    recorded = rec.get("jsonl") if isinstance(rec, dict) else None
    got = _read_set(setname, data, base, recorded, private, {} if cache is None else cache)
    if got is None:
        return None
    rows = rows_for_set(got[0], got[1], setname, S.get("predictions"), S.get("qfrag"))
    return {"cols": COLS, "rows": rows} if rows else None


def has_charges(J: dict) -> bool:
    """Whether a result carries model fragment charges anywhere (``qfrag``): decided per model, not per set."""
    return any((S or {}).get("qfrag") for S in (J.get("sets") or {}).values())


def private_points(path: str, runs: Optional[str] = None) -> dict:
    """Internal-only sets at a user-supplied local path, read live and never embedded.

    ``path`` holds canonical-schema ``<set>.jsonl`` + ``<set>.groups.json`` (directly or under ``canonical/``);
    model predictions come from ``lrbench eval --save-predictions`` results in ``<path>/runs/*.json`` and in ``runs``
    whose sets have the same names. Returns ``{"cols": COLS, "sets": [...], "models": {label: {set: rows}}, "ref": {set: rows}}``.
    """
    names = sorted({os.path.basename(p)[: -len(".jsonl")]
                    for p in glob.glob(os.path.join(path, "*.jsonl")) + glob.glob(os.path.join(path, "canonical",
                                                                                               "*.jsonl"))})
    cache: dict = {}
    out = {"cols": COLS, "sets": [], "models": {}, "ref": {}}
    for s in names:
        got = _read_set(s, path, None, None, None, cache)
        if got is None:
            continue
        rows = rows_for_set(got[0], got[1], s)
        if rows:
            out["sets"].append(s)
            out["ref"][s] = rows
    files = sorted(glob.glob(os.path.join(path, "runs", "*.json")))
    if runs:
        files += sorted(glob.glob(os.path.join(runs, "*.json")))
    for p in files:
        try:
            with open(p) as f:
                J = json.load(f)
        except Exception:
            continue
        for s in out["sets"]:
            S = (J.get("sets") or {}).get(s)
            if not S or not S.get("predictions"):
                continue
            got = _read_set(s, path, None, None, None, cache)
            out["models"].setdefault(J.get("label", os.path.basename(p)[:-5]), {})[s] = \
                rows_for_set(got[0], got[1], s, S.get("predictions"), S.get("qfrag"))
    return out


def attach(d: dict, J: dict, data: Optional[str], base: Optional[str], private: Optional[str],
           cache: Optional[dict] = None) -> dict:
    """Add ``ionct`` point tables to one distilled result (``lrbench.diagnostics.distil``) and its ``charges`` flag."""
    cache = {} if cache is None else cache
    d["charges"] = has_charges(J)
    for s in d.get("sets", {}):
        pts = points(J, s, data=data, base=base, private=private, cache=cache)
        if pts:
            d["sets"][s]["ionct"] = pts
    return d


def table(rows_by_set: Dict[str, dict]) -> List[dict]:
    """Rows as dicts (for tests and quick inspection)."""
    return [dict(zip(v["cols"], r)) for v in rows_by_set.values() for r in v["rows"]]
