"""lrbench command line: ``eval``, ``compare``, ``perf``, ``sets``.

Examples::

    lrbench eval --model model.gguf --sets dimers,families_fixedq --data ./lr-sets --out model.json
    lrbench eval --model ase:ase.calculators.lj:LennardJones?sigma=3.0 --sets trimers --out lj.json --equivariance
    lrbench compare model.json baseline.json --html report.html --figdir figs/
    lrbench perf --model pet-omol-s.gguf --boxes water3k,hca --box water3k=water-M.extxyz --box hca=min.extxyz
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from typing import Dict, List

import numpy as np

from . import __version__
from .backends import has_frag_charges, make_backend
from .data import (ALL_SETS, UNSCORABLE_LEVELS, available_sets, base_id, default_model_level, load_set, needed_terms,
                   record_level, subsample, term_structure)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _file_sha256(path: str, limit: int = 4 << 30):
    if not os.path.isfile(path) or os.path.getsize(path) > limit:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _model_path(spec: str) -> str:
    return spec.split(":", 1)[1] if spec.startswith("mlipcpp-capi:") else spec


def _formal_for(spec: str, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "from-data":
        return True
    # auto: only models that use formal charges (GGUF metadata pet.plato_lr.frag_charges)
    return has_frag_charges(_model_path(spec))


def _backend(spec: str, args, formal: bool):
    return make_backend(spec, device=args.device, formal=formal, tf32=args.tf32, mlipcpp_bin=args.mlipcpp_bin,
                        mlipcpp_lib=args.mlipcpp_lib, launcher=args.launcher, chunk=args.chunk)


def _predict_set(be, ds) -> Dict[str, dict]:
    terms = needed_terms(ds)
    structs = [term_structure(ds, t) for t in terms]
    if hasattr(be, "bind"):  # the multipole oracle reads fragment names / moments from the records
        be.bind(ds)
    preds = be.predict(structs)
    return dict(zip(terms, preds))


def _fragment_charge_sums(ds, preds, s) -> dict:
    """{record id: [[q_frag0, q_frag1, ...] per member]} from the model's per-atom charges (latent or equilibrated) on
    every record with fragment_id; empty if the model returns no charges (used by `lrbench emulation`)."""
    out = {}
    for r in ds.records:
        fid = r.get("fragment_id")
        if fid is None:
            continue
        fid = np.asarray(fid); nf = int(fid.max()) + 1
        per = []
        for m in preds:
            q = (m[s].get(r["id"]) or {}).get("charges")
            per.append(None if q is None else [float(np.asarray(q, float)[fid == k].sum()) for k in range(nf)])
        if any(p is not None for p in per):
            out[r["id"]] = per
    return out


def element_offsets(data, preds, mode: str):
    """Per-element energy offsets b_Z with E_ref ~ E_model + sum_Z n_Z b_Z, least squares over every scored record
    (not flagged, not a partner copy) of every set, on the member-mean energy. Models trained on reference-subtracted
    energies (PET-OMol-S: pet.composition_values of a few eV) are otherwise ~keV/atom off in absolute terms and in
    composition-changing differences; balanced differences (E_int, E3, fixed-composition dE/ddE) are unaffected."""
    if mode == "none":
        return None
    if mode != "fit":
        J = json.load(open(mode))  # {element: eV} (any level) or {level: {element: eV}}
        nested = all(isinstance(v, dict) for v in J.values())
        return {"mode": mode, "by_level": {k: {"offsets_eV": v} for k, v in J.items()} if nested else
                {"*": {"offsets_eV": J}}}
    by = {}  # level -> (rows, y)
    for s, ds in data.items():
        for r in ds.records:
            if r.get("energy") is None or ds.flagged(r["id"]) or ds.partner_copy(r["id"]) or not ds.energy_ok(r["id"]):
                continue
            e = [m[s].get(r["id"], {}).get("energy") for m in preds]
            if any(v is None for v in e):
                continue
            c = {}
            for z in r["species"]:
                c[z] = c.get(z, 0) + 1
            rows, y = by.setdefault(record_level(r), ([], []))
            rows.append(c)
            y.append(r["energy"] - float(np.mean(e)))
    if not by:
        return None
    res_ = {"mode": "fit", "by_level": {}, "note": "fitted in-sample, separately per label level (never pooled); "
            "residual includes charge-state and level effects"}
    for lv, (rows, y) in by.items():
        E = sorted({z for c in rows for z in c})
        X = np.array([[c.get(z, 0) for z in E] for c in rows], float)
        b, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
        res = np.array(y) - X @ b
        res_["by_level"][lv] = {"offsets_eV": dict(zip(E, map(float, b))), "n_records": len(y),
                                "residual_mae_meV_atom": float(1e3 * np.mean(np.abs(res) / X.sum(1))),
                                "residual_max_abs_eV": float(np.abs(res).max())}
    return res_


def apply_offsets(data, preds, off):
    L = off["by_level"]
    for m in preds:
        for s, ds in data.items():
            for t, p in m[s].items():
                if p.get("energy") is not None:
                    lv = record_level(ds.by_id[base_id(t)])
                    b = (L.get(lv) or L.get("*") or {}).get("offsets_eV", {})
                    p["energy_raw"] = p["energy"]
                    p["energy"] = p["energy"] + sum(b.get(z, 0.0) for z in term_structure(ds, t).species)


def level_check(data, members, model_level, allow) -> dict:
    """Refuse energy metrics that compare a model trained at one level with labels at another (never pool levels)."""
    lv = model_level or default_model_level(members[0])
    out = {"model_level": lv, "model_level_source": "--model-level" if model_level else "inferred from the spec",
           "label_levels": {s: ds.levels() for s, ds in data.items()}, "mismatch": {}}
    if lv in UNSCORABLE_LEVELS:
        warnings.warn("model training level unknown (pass --model-level); energies are scored but not level-checked")
        out["warning"] = "model level unknown: energy metrics not level-checked"
        return out
    for s, levels in out["label_levels"].items():
        bad = [x for x in levels if x != lv and x not in UNSCORABLE_LEVELS]
        if bad:
            out["mismatch"][s] = bad
    if out["mismatch"] and not allow:
        raise SystemExit(f"lrbench eval: model level {lv!r} differs from the label level(s) {out['mismatch']}; "
                         "energy metrics across levels are refused (use forces / same-level differences, or pass "
                         "--allow-level-mismatch to score anyway; the mismatch is then recorded everywhere)")
    return out


def cmd_sets(args):
    for s in available_sets(args.data, args.revision):
        print(s)


def cmd_eval(args):
    from .metrics import Preds, SetScorer
    members = ([args.model] if args.model else []) + [m for m in (args.ensemble or "").split(",") if m]
    if not members:
        sys.exit("lrbench eval: give --model and/or --ensemble")
    sets, seen = [], set()  # "all" expands in place, so --sets all,pbe0mbd_subset works
    for s in args.sets.split(","):
        for x in (ALL_SETS if s == "all" else [s] if s else []):
            if x not in seen:
                seen.add(x); sets.append(x)
    t0 = time.time()
    data = {s: subsample(load_set(s, args.data, args.revision), args.max_per_set, args.seed) for s in sets}
    lvc = level_check(data, members, args.model_level, args.allow_level_mismatch)
    out = {"lrbench_version": __version__, "level": lvc, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "host": platform.node(), "argv": sys.argv[1:], "label": args.label or os.path.basename(members[0]),
           "members": [], "sets": {}, "notes": []}
    preds: List[Dict[str, Dict[str, dict]]] = []  # member -> set -> term -> pred
    backends = []
    for spec in members:
        formal = _formal_for(spec, args.formal_charges)
        be = _backend(spec, args, formal)
        backends.append(be)
        per = {}
        for s, ds in data.items():
            t1 = time.time()
            per[s] = _predict_set(be, ds)
            print(f"[lrbench] {os.path.basename(spec)} {s}: {len(per[s])} structures in {time.time() - t1:.1f} s",
                  file=sys.stderr, flush=True)
        preds.append(per)
        p = _model_path(spec)
        out["members"].append({"spec": spec, "path": os.path.abspath(p) if os.path.exists(p) else p,
                               "sha256": _file_sha256(p) if not args.no_hash else None,
                               "formal_charges": formal, **be.info()})
        if not be.passes_charge_spin:
            out["notes"].append(f"{spec}: backend passes no charge/spin; charged structures are evaluated neutral")
        # A model that fails on *some* structures is the dangerous case, because the survivors still produce a
        # full-looking table. The failures are rarely random: they fall on the biggest, most separated geometries,
        # which are precisely what the tail metrics are made of, so the model gets scored on its short range and
        # reported as though it had been scored on everything. Say so loudly, once, per member and per set.
        for s_, per_set in per.items():
            n_bad = sum(1 for p_ in per_set.values() if p_.get("energy") is None)
            if n_bad:
                frac = n_bad / max(1, len(per_set))
                msg = (f"{spec}: {n_bad} of {len(per_set)} predictions failed on {s_} ({frac:.0%}); those "
                       "structures are dropped from every metric, so coverage is not what the group n implies")
                out["notes"].append(msg)
                warnings.warn(msg)
                err = next((p_.get("error") for p_ in per_set.values() if p_.get("error")), None)
                if err:
                    out["notes"].append(f"{spec}: first error on {s_}: {err}")
    off = element_offsets(data, preds, args.element_offsets)
    if off:
        apply_offsets(data, preds, off)
        out["element_offsets"] = off
        out["notes"].append(f"per-element energy offsets ({off['mode']}) added to every model energy; affects only "
                            "absolute energies and composition-changing differences")
    if lvc["mismatch"]:
        out["notes"].insert(0, f"LEVEL MISMATCH (allowed): model {lvc['model_level']} vs labels {lvc['mismatch']}; "
                               "energy metrics compare different levels of theory")
    norm = None
    if args.normalise:
        from .gguf_tools import les_gate
        gates = [les_gate(_model_path(m)) for m in members]
        norm = dict(floor_meV=args.eps_floor, temperature=args.temperature, distance_unit=args.distance_unit,
                    energy_unit=args.energy_unit, gates=gates, frag=[has_frag_charges(_model_path(m)) for m in members])
        out["normalise"] = {k: v for k, v in norm.items()}
    for s, ds in data.items():
        P = Preds([m[s] for m in preds])
        if s == "props":  # electrostatic properties (lrbench.props): charges/dipoles/field response, no energies
            from .props import score as props_score
            out["sets"][s] = props_score(ds, P)
        else:
            out["sets"][s] = SetScorer(ds, P, n_boot=args.n_boot, conformal_on=args.conformal, normalise=norm).run()
        if args.save_predictions:
            out["sets"][s]["predictions"] = {t: [None if m[s][t].get("energy") is None else m[s][t]["energy"]
                                                 for m in preds] for t in preds[0][s]}
            qf = _fragment_charge_sums(ds, preds, s)
            if qf:
                out["sets"][s]["qfrag"] = qf
    if len(members) > 1:
        out["ensemble"] = {"n_members": len(members), "prediction": "mean of member energies per structure; "
                           "every difference is the mean over members of the member's difference; spread = member std "
                           "(ddof 1) used as sigma for normalised conformal scores"}
        ggufs = [_model_path(m) for m in members]
        if all(g.endswith(".gguf") for g in ggufs) and not args.no_weight_average:
            from .gguf_tools import average
            avg = os.path.join(tempfile.mkdtemp(prefix="lrbench-avg-"), "weight_average.gguf")
            ok, why = average(ggufs, avg)
            out["ensemble"]["weight_average"] = {"done": ok, "reason": why}
            if ok:
                be = _backend(avg, args, _formal_for(members[0], args.formal_charges))
                wp = [{s: _predict_set(be, ds) for s, ds in data.items()}]
                if off:
                    apply_offsets(data, wp, off)
                wa = {s: SetScorer(ds, Preds([wp[0][s]]), n_boot=args.n_boot).run() for s, ds in data.items()}
                out["ensemble"]["weight_average"]["sets"] = wa
                out["ensemble"]["weight_average"]["caveat"] = ("averaging independently trained seeds' weights is "
                                                              "meaningful only if they share a basin")
        else:
            out["ensemble"]["weight_average"] = {"done": False, "reason": "not all members are GGUF files" if not
                                                 args.no_weight_average else "disabled (--no-weight-average)"}
    if args.equivariance:
        from .equivariance import check, sample
        pool = [term_structure(ds, r["id"]) for ds in data.values() for r in ds.records
                if not ds.flagged(r["id"])]
        smp = sample(pool, args.equivariance_n, args.seed)
        out["equivariance"] = check(backends[0], smp, n_rot=args.n_rot, seed=args.seed)
        out["equivariance"]["member"] = members[0]
        out["equivariance"]["pbc_checked"] = any(s.cell is not None and any(s.pbc) for s in smp)
        if not out["equivariance"]["pbc_checked"]:
            out["equivariance"]["pbc_note"] = "no periodic structure in the sample: wrap/image checks not run"
    if args.rot_eint and "dimers" in data:
        from .roteint import run as rot_run
        out["rotational_eint"] = rot_run(backends[0], data["dimers"], [float(x) for x in args.rot_eint.split(",")],
                                         n_rot=args.rot_eint_n, seed=args.seed, max_scans=args.rot_eint_max_scans)
        out["rotational_eint"]["member"] = members[0]
    out["data"] = {s: ds.source for s, ds in data.items()}
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    print(f"[lrbench] wrote {args.out} ({out['wall_s']} s)", file=sys.stderr)
    from .report import headline
    print(headline(out))


def read_structures(path: str) -> list:
    """extxyz -> Structures, with the charge / spin_multiplicity (else spin, multiplicity) header keys."""
    from ase.io import read
    from .data import Structure
    out = []
    for i, a in enumerate(read(path, ":")):
        sp = a.info.get("spin_multiplicity", a.info.get("spin", a.info.get("multiplicity", 1)))
        out.append(Structure(f"{os.path.basename(path)}:{i}", list(a.get_chemical_symbols()), a.positions.copy(),
                             int(a.info.get("charge", 0)), int(sp), a.cell.array.copy() if a.pbc.any() else None,
                             tuple(bool(x) for x in a.pbc),
                             a.arrays["formal_charges"].astype(float).copy() if "formal_charges" in a.arrays else None))
    return out


def cmd_predict(args):
    """Any model spec on an extxyz -> mlipcpp-eval-style JSON ({"frames": [{energy, forces, charges, dipole}]}), so
    the program's scripts that read mlipcpp-eval output (eps_monitor.py, figures/scripts) take external models."""
    structs = read_structures(args.xyz)
    be = _backend(args.model, args, args.formal_charges == "from-data")
    t0 = time.time()
    preds = be.predict(structs)
    wall = time.time() - t0
    frames = []
    for p in preds:
        f = {"energy": p["energy"], "forces": None if p.get("forces") is None else np.asarray(p["forces"]).tolist()}
        for k in ("charges", "dipole"):
            if p.get(k) is not None:
                f[k] = np.asarray(p[k]).tolist()
        if p.get("error"):
            f["error"] = p["error"]
        frames.append(f)
    out = {"model": args.model, "level": args.model_level or default_model_level(args.model), "xyz": args.xyz,
           "n_frames": len(frames), "n_errors": sum("error" in f for f in frames), "wall_s": round(wall, 2),
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
           "backend": be.info(), "frames": frames}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(_jsonable(out), fh)
    print(f"[lrbench] {args.model}: {len(frames)} frames ({out['n_errors']} errors) in {wall:.1f} s -> {args.out}",
          file=sys.stderr)


def cmd_suite(args):
    from .suite import run_suite
    extra = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    S = run_suite(args.models, args.outdir, sets=args.sets, data=args.data, device=args.device, extra=extra,
                  pythons=dict(kv.split("=", 1) for kv in args.python), include=args.include,
                  dry_run=args.dry_run, sbatch_header=args.sbatch_header)
    for r in S["runs"]:
        print(f"[lrbench suite] {r['name']}: {r['status']}" + (f" ({r['reason']})" if r.get("reason") else ""),
              file=sys.stderr)
    if S.get("combined_md"):
        print(open(S["combined_md"]).read())


def cmd_models(args):
    from .models import MODELS, requirements_md
    if args.md:
        print(requirements_md())
    else:
        for m in MODELS.values():
            print(f"{m.name:15s} {m.level:22s} env={m.env:6s} {m.spec}")


def cmd_emulation(args):
    from . import emulation as EM
    rows, energy = [], {}
    for p in args.results:
        R = json.load(open(p)); lab = R.get("label_short") or R.get("label")
        rows += EM.rows_from_result(R, args.data, lab)
        energy[lab] = EM.energy_scores(R, args.data)
    for kv in args.charges:
        tag, path = kv.split("=", 1)
        rows += EM.rows_from_chemiscope(path, tag)
    out = {"definition": EM.__doc__, "tol_e": args.tol, "frac_tol_e": args.frac_tol,
           "fragment_charges": EM.score_rows(rows, args.tol, args.frac_tol), "energy": energy}
    if args.sizecons and args.sizecons_dft:
        S = json.load(open(args.sizecons))["models"]
        D = [dict(r, sys=r["sys"]) for r in json.load(open(args.sizecons_dft))["scans"] if r.get("state", "singlet") == "singlet"]
        out["size_consistency_vs_dft"] = {m: EM.size_consistency(v.get("sizecons", []), D) for m, v in S.items()}
    with open(args.out, "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    if args.fig:
        with open(args.fig, "w") as f:
            json.dump(_jsonable({"note": "fragment-charge rows (e): model vs DFT Loewdin vs formal", "rows": rows}), f)
    for m, v in out["fragment_charges"].items():
        k = v.get("kind", {})
        print(f"[lrbench emulation] {m}: charges pass vs DFT integer-like "
              f"{(k.get('integer-like') or {}).get('pass_rate_vs_dft')}, fractional/CT {(k.get('fractional/CT') or {}).get('pass_rate_vs_dft')}",
              file=sys.stderr)


def cmd_compare(args):
    from .report import compare
    md = compare(args.results, html=args.html, figdir=args.figdir, md_out=args.md, emulation=args.emulation)
    print(md)
    for mode in [m for m in ("split", "seen") if getattr(args, m)]:
        _rescored_table(args, mode)


def _rescored_table(args, mode):
    """Second (and third) table rescored from saved predictions: held-out pairs (--split) / unseen records (--seen)."""
    from .report import compare
    from .split import rescore, unseen_ids
    tmp = tempfile.mkdtemp(prefix="lrbench-split-")
    side = args.split_side if mode == "split" else "unseen"
    ids = None
    paths = []
    for p in args.results:
        R = json.load(open(p))
        if not any("predictions" in b for b in R.get("sets", {}).values()):
            print(f"[lrbench] {p}: no saved predictions, left out of the {side} table", file=sys.stderr)
            continue
        q = os.path.join(tmp, f"{len(paths)}.json")
        if mode == "seen":
            data = args.data or next(iter(R["data"].values()))["data"]
            ids = ids or unseen_ids(args.seen, data, [s for s in R["sets"]] + ["dimers", "trimers", "families_full",
                                                                                 "families_corrected", "families_fixedq"])
        json.dump(_jsonable(rescore(R, args.split, side, args.data, ids=ids)), open(q, "w"))
        paths.append(q)
    stem = lambda f: None if not f else f.rsplit(".", 1)[0] + f".{side}." + f.rsplit(".", 1)[1]
    md2 = compare(paths, html=stem(args.html), md_out=stem(args.md))
    print(f"\n## {side} only ({args.split if mode == 'split' else args.seen})\n\n" + md2)


BOX_DEFAULTS = {}  # no built-in paths: give a box with --box NAME=FILE (generated water boxes need none)
SCAN = [1000, 10000, 50000, 100000]


def cmd_perf(args):
    from .perf import read_box, run_perf, water_box
    paths = dict(BOX_DEFAULTS)
    for kv in args.box or []:
        k, v = kv.split("=", 1)
        paths[k] = v
    boxes = []
    for name in [b for b in args.boxes.split(",") if b]:
        if name in paths:
            p = os.path.expanduser(paths[name])
            if not os.path.exists(p):
                warnings.warn(f"box {name}: {p} not found; skipped (pass --box {name}=PATH)")
                continue
            boxes.append(read_box(p, name))
        elif name == "scan":
            boxes += [water_box(n, seed=args.seed) for n in SCAN]
        elif name.startswith("water") and name[5:].rstrip("k").isdigit():
            n = int(name[5:].rstrip("k")) * (1000 if name.endswith("k") else 1)
            boxes.append(water_box(n, seed=args.seed))
        elif os.path.exists(name):
            boxes.append(read_box(name, os.path.basename(name)))
        else:
            warnings.warn(f"unknown box {name!r}; skipped")
    be = _backend(args.model, args, False)
    frag = [int(x) for x in args.frag_sizes.split(",")] if args.frag else []
    res = run_perf(be, boxes, args.warmup, args.n, args.md_steps, args.model, frag)
    res.update(lrbench_version=__version__, created=datetime.datetime.now().isoformat(timespec="seconds"),
               model=args.model, backend=be.info(), env={k: os.environ.get(k) for k in
                                                          ("MLIPCPP_RAGGED", "MLIPCPP_TF32", "GGML_CUDA_NO_TF32",
                                                           "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")},
               )
    with open(args.out, "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    from .report import perf_table
    print(perf_table(res["rows"]))
    print(f"[lrbench] wrote {args.out}", file=sys.stderr)


def cmd_toy(args):
    from .toy import LJ_SPEC, build
    d = build(args.outdir)
    print(f"[lrbench] wrote the toy set to {d}\n"
          f"[lrbench] lrbench eval --model \"{LJ_SPEC}\" --sets toy --data {d} --out toy.json", file=sys.stderr)


def cmd_viz(args):
    from .viz import parse_models, run
    models = parse_models(args.model, lambda m: _backend(m, args, _formal_for(m, args.formal_charges)))
    d = run(args.source, args.out, args.compare or [], models, args.max_n)
    print(f"[lrbench] wrote {args.out}: {len(d['structures'])} structures, properties "
          + ", ".join(d["properties"]), file=sys.stderr)


def _common(p):
    p.add_argument("--device", default="cpu", help="cpu | cuda (mlip.cpp backend / torch device)")
    p.add_argument("--tf32", action="store_true", help="mlip.cpp TF32 matmuls via MLIPCPP_TF32=1 (never "
                   "NVIDIA_TF32_OVERRIDE); default fp32 with GGML_CUDA_NO_TF32=1")
    p.add_argument("--mlipcpp-bin", help="mlipcpp-eval binary (else $LRBENCH_MLIPCPP_EVAL, PATH)")
    p.add_argument("--mlipcpp-lib", help="libmlipcpp_c.so for the C API path (else $MLIPCPP_C_LIB)")
    p.add_argument("--launcher", default="", help="prefix for the mlipcpp-eval command, e.g. 'srun -n1'")
    p.add_argument("--chunk", type=int, default=0, help="frames per mlipcpp-eval call (0 = all)")
    p.add_argument("--seed", type=int, default=0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lrbench", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="score a model (or ensemble) on the canonical long-range sets")
    p.add_argument("--model", help="name (lrbench models) | path.gguf | mlipcpp-capi:path.gguf | model.pt | ase:module:Factory?kw=v | oracle:multipole?order=2&ind=1&moments=frags.json")
    p.add_argument("--ensemble", help="comma-separated further members (scored as one prediction-averaged model)")
    p.add_argument("--sets", default="dimers,families_fixedq,trimers", help="comma list or 'all'")
    p.add_argument("--data", default="hf", help="'hf' (the canonical HF dataset, $LRBENCH_HF_REPO) or a local directory")
    p.add_argument("--revision", help="HF dataset revision")
    p.add_argument("--out", required=True)
    p.add_argument("--label", help="model label for tables (default: the model file name)")
    p.add_argument("--formal-charges", default="auto", choices=["auto", "none", "from-data"],
                   help="auto: only for GGUFs with pet.plato_lr.frag_charges")
    p.add_argument("--model-level", help="training level of the model (default: wB97M-V/def2-TZVPD for .gguf/.pt, "
                   "unknown for ase: specs)")
    p.add_argument("--allow-level-mismatch", action="store_true", help="score energies against labels at another "
                   "level anyway (recorded in results.json and compare tables)")
    p.add_argument("--element-offsets", default="fit", help="fit | none | offsets.json ({element: eV}): per-element "
                   "energy offsets for models trained on reference-subtracted energies")
    p.add_argument("--normalise", action="store_true", help="physics-normalised errors (ERM_priors.md): dimer tail "
                   "eps(r), k_BT errors, normalised ddE / E3 / dipoles, g lambda^2 - 1 for LES models")
    p.add_argument("--temperature", type=float, default=300.0, help="K, for errors in k_BT")
    p.add_argument("--distance-unit", default="A", choices=["A", "bohr"])
    p.add_argument("--energy-unit", default="meV", choices=["meV", "eV", "Eh", "kBT"],
                   help="unit of absolute errors in the normalised block (all other blocks stay meV)")
    p.add_argument("--eps-floor", type=float, default=0.1, help="meV; normalising scales below this are not scored")
    p.add_argument("--equivariance", action="store_true")
    p.add_argument("--equivariance-n", type=int, default=8)
    p.add_argument("--n-rot", type=int, default=3)
    p.add_argument("--rot-eint", default="", help="comma list of d_min (A), e.g. 8,12,16,24: rotational spread of "
                   "dimer E_int at those separations (lrbench.roteint; needs the dimers set)")
    p.add_argument("--rot-eint-n", type=int, default=8, help="random rotations per scan and distance (+ identity)")
    p.add_argument("--rot-eint-max-scans", type=int, default=0, help="subsample scans (0 = all)")
    p.add_argument("--max-per-set", type=int, default=0, help="subsample whole scans/groups to about N records")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--conformal", action="store_true", help="split-conformal / CQR even without an ensemble")
    p.add_argument("--no-weight-average", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--no-hash", action="store_true", help="skip the model sha256")
    _common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("compare", help="tables and figures from several result JSONs")
    p.add_argument("results", nargs="+")
    p.add_argument("--html")
    p.add_argument("--md")
    p.add_argument("--figdir")
    p.set_defaults(func=cmd_compare)

    p.add_argument("--split", help="pair-level split JSON (heldout_point_ids): also write a held-out-only table "
                   "(<md>.heldout.md / .html), rescored from each result's saved predictions")
    p.add_argument("--split-side", default="heldout")
    p.add_argument("--emulation", help="lrbench emulation output JSON: adds emulation / physics score rows")
    p.add_argument("--seen", help="seen_by JSON (tools/seen_by.py): also write an unseen-only table (<md>.unseen.md)")
    p.add_argument("--data", help="dataset dir for --split rescoring (default: each result's recorded data path)")

    p = sub.add_parser("emulation", help="DFT-emulation scores: fragment charges vs DFT (not formal), emulation vs "
                       "physics E_int scores, size consistency vs DFT's own decay (lrbench.emulation)")
    p.add_argument("--results", action="append", default=[], help="eval result JSON with saved predictions (qfrag used "
                   "for charges when present)")
    p.add_argument("--charges", action="append", default=[], help="TAG=chemiscope.json(.gz) from lrbench viz (q_latent/TAG)")
    p.add_argument("--sizecons", help="tools/scans.py analyse output (models -> sizecons rows)")
    p.add_argument("--sizecons-dft", help="DFT size-consistency rows (tools/p2_parse.py output, 'scans')")
    p.add_argument("--data", help="dataset dir for the results' sets")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--frac-tol", type=float, default=0.02)
    p.add_argument("--out", required=True)
    p.add_argument("--fig", help="figure-ready per-row JSON")
    p.set_defaults(func=cmd_emulation)

    p = sub.add_parser("perf", help="timing on standard boxes")
    p.add_argument("--model", required=True)
    p.add_argument("--boxes", default="scan", help="scan (generated water at 1k/10k/50k/100k atoms), waterN "
                   "(generated, N atoms), or any name given with --box")
    p.add_argument("--box", action="append", help="NAME=PATH.extxyz: a structure file to time, e.g. "
                   "--box hca=min.extxyz. Generated water boxes need no file.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--md-steps", type=int, default=0, help="also time VelocityVerlet steps (C API / ASE backends)")
    p.add_argument("--frag", action="store_true", help="time the host fragment-charge term (mlipcpp-eval)")
    p.add_argument("--frag-sizes", default="3000,30000")
    p.add_argument("--out", default="perf.json")
    _common(p)
    p.set_defaults(func=cmd_perf)

    p = sub.add_parser("viz", help="chemiscope JSON(.gz) of a set (ours or external) with the explorer schema")
    p.add_argument("--source", required=True, help="hf://<repo>/<config> | DIR:<config> | "
                   "file.jsonl | file.extxyz | https://...")
    p.add_argument("--out", required=True, help="x.json or x.json.gz")
    p.add_argument("--compare", action="append", help="further source or chemiscope .json[.gz] to merge (repeatable)")
    p.add_argument("--model", action="append", help="TAG=spec[,spec...] (seed set -> mean and _sd) or a bare spec; "
                   "repeatable; columns <q>/TAG")
    p.add_argument("--formal-charges", default="auto", choices=["auto", "none", "from-data"])
    p.add_argument("--max-n", type=int, default=0)
    _common(p)
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("predict", help="any model on an extxyz -> mlipcpp-eval-style JSON (frames)")
    p.add_argument("--model", required=True)
    p.add_argument("xyz")
    p.add_argument("out")
    p.add_argument("--model-level")
    p.add_argument("--formal-charges", default="none", choices=["none", "from-data"])
    _common(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("suite", help="run eval for several models (each in its own environment) and write one "
                       "combined table; e.g. --models external = so3lr,so3lr-tauri,mace-h,mace-polar")
    p.add_argument("--models", default="external", help="comma list of names, or a group: external, all-external")
    p.add_argument("--include", action="append", default=[], help="existing result JSON to add to the combined "
                   "table (e.g. our own models); repeatable")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sets", default="dimers,families_fixedq,trimers")
    p.add_argument("--data", default="hf")
    p.add_argument("--python", action="append", default=[], help="ENV=/path/to/python for an environment "
                   "(so3lr, mace); else $LRBENCH_PY_<ENV>, else this interpreter")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="write run_<model>.sh, suite.sbatch and suite.json "
                   "without running anything")
    p.add_argument("--sbatch-header", help="file whose #SBATCH lines go at the top of suite.sbatch")
    p.add_argument("--continue-on-error", action="store_true", default=True)
    p.add_argument("eval_args", nargs=argparse.REMAINDER, help="after --: extra arguments for every lrbench eval "
                   "(e.g. -- --normalise --conformal)")
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("models", help="list the named models and what each needs to run")
    p.add_argument("--md", action="store_true", help="markdown requirements table")
    p.set_defaults(func=cmd_models)

    from .priors.run import add_parser as _priors_parser
    _priors_parser(sub, _common)

    p = sub.add_parser("toy", help="write the self-contained Lennard-Jones toy set (no download, for the quickstart)")
    p.add_argument("outdir")
    p.set_defaults(func=cmd_toy)

    p = sub.add_parser("sets", help="list the available sets")
    p.add_argument("--data", default="hf")
    p.add_argument("--revision")
    p.set_defaults(func=cmd_sets)

    args = ap.parse_args(argv)
    _resolve_names(args)
    args.func(args)


def _resolve_names(args):
    """Short model names (lrbench.models: so3lr, so3lr-tauri, mace-h, mace-polar, ...) -> adapter specs. The name
    becomes the default --label, so tables show it."""
    from .models import resolve
    m = getattr(args, "model", None)
    if isinstance(m, str):
        r = resolve(m)
        if r != m and hasattr(args, "label") and not args.label:
            args.label = m
        args.model = r
    if getattr(args, "ensemble", None):
        args.ensemble = ",".join(resolve(x) for x in args.ensemble.split(",") if x)


if __name__ == "__main__":
    main()
