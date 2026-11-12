"""``lrbench priors``: build the context (backend, dimer set, optional probe files), run the checks, write JSON + md.

Optional probe inputs (extxyz; builders in tools/priors_*.py):
  --ipea FILE        vertical IP/EA probes: info charge, spin_multiplicity, frag, IP_eV, EA_eV, mult_Nm1, mult_Np1, flags
  --sep-frames FILE  separated / single-component frames: info charge, spin_multiplicity, E_dft, ref_offset, stratum
  --seen FILE        seen_by JSON (tools/seen_by.py; {"seen_by": {"dimers": {id: [sources]}}})
  --train-coverage   JSON list / JSONL of training rows with g (or d) and qa/qb (or cls)
  --baseline FILE    an earlier priors JSON (paired checks, e.g. IP damage vs PET-OMol-S)
  --moments FILE     fragment DFT moments (frags.json of the dimer build; $LRBENCH_FRAG_MOMENTS), for the
                     oracle-scored orientation checks
"""
from __future__ import annotations

import datetime
import json
import os
import sys


from ..data import Structure, default_model_level, load_set
from .core import REGISTRY, Ctx, run

SYMBOL = {"pass": "pass", "fail": "**FAIL**", "warn": "warn", "not_applicable": "n/a", "not_evaluable": "n/e"}


def _read(path):
    from ase.io import read
    return read(path, ":")


def _struct(a, key):
    sp = a.info.get("spin_multiplicity", a.info.get("spin", 1))
    return Structure(key, list(a.get_chemical_symbols()), a.positions.copy(), int(a.info.get("charge", 0)), int(sp))


def load_ipea(path):
    out = []
    for i, a in enumerate(_read(path)):
        f = lambda k: None if a.info.get(k) in (None, "None", "nan") else float(a.info[k])
        out.append({"frag": str(a.info.get("frag", i)), "structure": _struct(a, f"ipea:{a.info.get('frag', i)}"),
                    "IP_eV": f("IP_eV"), "EA_eV": f("EA_eV"),
                    "mult_states": {"Nm1": int(a.info["mult_Nm1"]) if "mult_Nm1" in a.info else None,
                                    "Np1": int(a.info["mult_Np1"]) if "mult_Np1" in a.info else None},
                    "flags": [x for x in str(a.info.get("flags", "")).split(";") if x]})
    return out


def load_sepframes(path):
    return [{"structure": _struct(a, f"sep:{i}"), "E_dft": float(a.info["E_dft"]),
             "ref_offset": float(a.info.get("ref_offset", 0.0)), "stratum": str(a.info.get("stratum", "separated"))}
            for i, a in enumerate(_read(path))]


def load_coverage(paths):
    rows = []
    for p in paths:
        if p.endswith(".jsonl"):
            rows += [json.loads(l) for l in open(p) if l.strip()]
        else:
            rows += json.load(open(p))
    return rows


def model_info(spec: str, backend):
    """(cutoff A or None, source, declared_local or None) from the C API / GGUF metadata."""
    cut, src, local = None, None, None
    lib, m = getattr(backend, "lib", None), getattr(backend, "m", None)
    if lib is not None and m is not None:
        import ctypes
        c = ctypes.c_float()
        try:
            lib.mlipcpp_model_get_cutoff.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
            if lib.mlipcpp_model_get_cutoff(m, ctypes.byref(c)) == 0 and c.value > 0:
                cut, src = float(c.value), "mlipcpp_model_get_cutoff"
        except AttributeError:
            pass
    path = spec.split(":", 1)[1] if spec.startswith("mlipcpp-capi:") else spec
    if path.endswith(".gguf") and os.path.exists(path):
        from ..backends import gguf_metadata
        md = gguf_metadata(path)
        if md:
            local = not any(("plato_lr" in k or "lrc_" in k or ".les" in k or "coulomb" in k.lower()) for k in md)
            if cut is None:
                for k in ("pet.cutoff", "pet.hypers.cutoff", "cutoff"):
                    if isinstance(md.get(k), (int, float)):
                        cut, src = float(md[k]), f"gguf {k}"
                        break
    return cut, src, local


def markdown(res: dict) -> str:
    L = [f"### {res['label']}: priors and assumptions", "",
         f"model `{res['model']}`, backend {res['backend'].get('backend')}, "
         f"{res['backend'].get('precision', '')}, {res['created']}", "",
         "| check | kind | outcome | result |", "|---|---|---|---|"]
    for n, r in res["checks"].items():
        L.append(f"| {n} | {r['kind']} | {SYMBOL.get(r['status'], r['status'])} | {r['summary'].replace('|', '/')} |")
    return "\n".join(L) + "\n"


def list_checks() -> str:
    """The registry as a markdown table: what each check asserts and how it decides."""
    from .checks import ORDER
    out = ["| check | kind | prior it asserts | pass / fail rule |", "|---|---|---|---|"]
    for n in ORDER:
        c = REGISTRY[n]
        cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        out.append(f"| `{c.name}` | {c.kind} | {cell(c.prior)} | {cell(c.rule)} |")
    return "\n".join(out)


def cmd_priors(args):
    if getattr(args, "list", False):
        print(list_checks())
        return
    missing = [f for f, v in (("--model", args.model), ("--out", args.out)) if not v]
    if missing:
        sys.exit(f"lrbench priors: {' and '.join(missing)} required (or pass --list)")
    from ..backends import make_backend
    names = [c for c in (args.checks.split(",") if args.checks != "all" else list(REGISTRY)) if c]
    from .checks import ORDER
    names = [n for n in ORDER if n in names] + [n for n in names if n not in ORDER]
    bad = [n for n in names if n not in REGISTRY]
    if bad:
        sys.exit(f"unknown checks {bad}; known: {', '.join(REGISTRY)}")
    be = make_backend(args.model, device=args.device, formal=False, tf32=args.tf32, mlipcpp_bin=args.mlipcpp_bin,
                      mlipcpp_lib=args.mlipcpp_lib, launcher=args.launcher, chunk=args.chunk)
    if hasattr(be, "keep_sqe"):  # mlipcpp-eval: latent charges are the conditioning input; sqe_charges the physical ones
        be.keep_sqe = True
        ctx_opts = {"charge_key": "charges_phys", "latent_key": "charges"}
    else:
        ctx_opts = {"charge_key": "charges"}
    ref = None
    if args.ref_model:
        ref = make_backend(args.ref_model, device=args.ref_device or args.device, formal=False, tf32=False,
                           mlipcpp_bin=args.mlipcpp_bin, mlipcpp_lib=args.mlipcpp_lib)
    ds = load_set("dimers", args.data) if args.data else None
    cut, src, local = model_info(args.model, be)
    if args.cutoff:
        cut, src = args.cutoff, "--cutoff"
    seen = None
    if args.seen:
        S = json.load(open(args.seen))
        seen = (S.get("seen_by") or S).get("dimers", {})
    ctx = Ctx(backend=be, label=args.label or os.path.basename(args.model), dimers=ds,
              model_level=args.model_level or default_model_level(args.model),
              allow_level_mismatch=args.allow_level_mismatch, ipea=load_ipea(args.ipea) if args.ipea else None,
              sepframes=load_sepframes(args.sep_frames) if args.sep_frames else None, seen=seen,
              train_coverage=load_coverage(args.train_coverage) if args.train_coverage else None, ref_backend=ref,
              baseline=json.load(open(args.baseline)) if args.baseline else None, cutoff=cut, declared_local=local,
              opts={"cutoff_source": src, "ref_model": args.ref_model, **ctx_opts})
    mom = args.moments or os.environ.get("LRBENCH_FRAG_MOMENTS")
    if mom:
        from ..oracle import load_frags
        ctx.moments = load_frags(mom)
    ctx.opts.update(orient_n=args.orient_n, orient_seed=args.orient_seed, scan_orients=args.scan_orientations,
                    orient_out=args.orient_out)
    if args.noise_rel is not None:
        ctx.noise_rel = args.noise_rel
    if args.max_pairs:
        from .checks import _templates
        ctx.opts["_tpl"] = _templates(ctx)[:args.max_pairs]
    res = {"label": ctx.label, "model": args.model, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "backend": be.info(), "model_level": ctx.model_level, "cutoff_A": cut, "cutoff_source": src,
           "declared_local": local, "data": ds.source if ds is not None else None,
           "inputs": {k: getattr(args, k) for k in ("ipea", "sep_frames", "seen", "train_coverage", "baseline",
                                                    "ref_model", "moments", "orient_n", "orient_seed",
                                                    "scan_orientations")},
           "env": {k: os.environ.get(k) for k in ("GGML_CUDA_NO_TF32", "MLIPCPP_PAD_EXACT", "MLIPCPP_TF32",
                                                   "MLIPCPP_RAGGED", "NVIDIA_TF32_OVERRIDE")}}
    res["checks"] = run(ctx, names, log=lambda s: print(s, file=sys.stderr, flush=True))
    res["model_calls"] = ctx.n_calls
    from ..cli import _jsonable
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(_jsonable(res), open(args.out, "w"), indent=1)
    md = markdown(res)
    if args.md:
        open(args.md, "w").write(md)
    print(md)


def add_parser(sub, common):
    p = sub.add_parser("priors", help="priors and assumptions suite: physical-prior / model / data checks with "
                       "pass-fail rules (separate from eval; eval outputs are unchanged)")
    p.add_argument("--list", action="store_true", help="print the checks and their pass/fail rules, then exit")
    p.add_argument("--model")
    p.add_argument("--label")
    p.add_argument("--checks", default="all", help="comma list or 'all' (lrbench priors --list)")
    p.add_argument("--data", default="hf", help="dataset dir or 'hf' for the canonical dimer set ('' = none)")
    p.add_argument("--out")
    p.add_argument("--md")
    p.add_argument("--model-level")
    p.add_argument("--allow-level-mismatch", action="store_true")
    p.add_argument("--ipea", help="IP/EA probe extxyz (tools/priors_ipea.py)")
    p.add_argument("--sep-frames", help="separated / single-component frames extxyz (tools/priors_sepframes.py)")
    p.add_argument("--seen", help="seen_by JSON")
    p.add_argument("--train-coverage", action="append", default=[], help="training-row metadata JSON / JSONL")
    p.add_argument("--baseline", help="priors JSON of a baseline model (paired checks)")
    p.add_argument("--ref-model", help="reference backend spec for the precision check")
    p.add_argument("--ref-device")
    p.add_argument("--cutoff", type=float, help="model cutoff (A) when the backend does not report one")
    p.add_argument("--noise-rel", type=float, help="relative float floor of total energies (default 1.2e-7, fp32)")
    p.add_argument("--max-pairs", type=int, default=0, help="limit the scan templates (smoke runs)")
    p.add_argument("--moments", help="fragment DFT moments, frags.json (oracle-scored orientation checks)")
    p.add_argument("--orient-n", type=int, default=128, help="random rigid orientations per pair (orientation_far)")
    p.add_argument("--orient-seed", type=int, default=2026)
    p.add_argument("--orient-out", help="extxyz of the worst geometry per pair and band (orientation_adversarial)")
    p.add_argument("--scan-orientations", type=int, default=0,
                   help="also run seam / energy_force / conservativeness on K random rigid orientations per pair")
    common(p)
    p.set_defaults(func=cmd_priors)
    return p
