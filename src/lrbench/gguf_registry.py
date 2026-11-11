"""Named mlip.cpp GGUF models, each bound to the mlip.cpp build it must be evaluated with.

A GGUF is not self-contained in two ways that matter for scoring:

  - **The build.** Long-range variants add metadata keys (``pet.plato_lr.lrc_qproj``, ``lrc_links``, ...) that only the
    branch that trained them reads. Another build loads the same file without complaint and silently ignores those
    keys, so the model it scores is not the model that was trained. Each entry therefore names the mlip.cpp commit
    (``build``); the binary is looked up from it, and a binary built from another commit is refused.
  - **Companion files.** A charge-head model points at further GGUFs by name, relative to its own directory
    (``pet.plato_lr.lrc_qmodel = q.gguf``, ``lrc_fmodel``, ``lrc_fmodel_minus``). They are hashed like the model, so
    a stale ``q.gguf`` that happens to sit next to the model is caught.

The registry is a JSON file (``configs/gguf_models.json``, or ``$LRBENCH_GGUF_REGISTRY``). Paths in it are written
against two environment variables, so the file names no machine:

  - ``LRBENCH_GGUF_ROOT``       directory with ``<NAME>/model.gguf`` and its companions;
  - ``LRBENCH_MLIPCPP_BUILDS``  directory with ``<commit>/bin/mlipcpp-eval`` and a ``COMMIT`` file beside it.

A registered name is used like any other model name, and every lrbench command takes it:

    lrbench eval --model ftq13 --sets all --data ./lr-sets --out ftq13.json

The member block of the result then carries ``gguf_registry``: the entry, the binary, its commit and the companion
hashes. ``python -m lrbench.gguf_registry check`` verifies every entry (hashes, companions, binary commit, metadata
keys); ``smoke`` evaluates one structure per model; ``add`` writes an entry from a model file and a build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

ENV_REGISTRY = "LRBENCH_GGUF_REGISTRY"
ENV_ALLOW = "LRBENCH_ALLOW_BINARY_MISMATCH"  # =1: record a binary/commit mismatch instead of refusing it
#: metadata keys that name a companion GGUF, relative to the model's directory
COMPANION_KEYS = ("pet.plato_lr.lrc_qmodel", "pet.plato_lr.lrc_fmodel", "pet.plato_lr.lrc_fmodel_minus")
#: metadata prefixes that are not read at inference (trainer bookkeeping, provenance)
_NOT_READ = ("general.", "train.", "tokenizer.", "GGUF.")


def _default_path() -> Optional[str]:
    p = os.environ.get(ENV_REGISTRY)
    if p:
        return p
    src = Path(__file__).resolve().parents[2] / "configs" / "gguf_models.json"  # a source checkout
    return str(src) if src.exists() else None


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path) as f:
        R = json.load(f)
    R.setdefault("models", {})
    R.setdefault("builds", {})
    R.setdefault("groups", {})
    return R


def registry(path: Optional[str] = None) -> dict:
    p = path or _default_path()
    return _load(p) if p and os.path.exists(p) else {"models": {}, "builds": {}, "groups": {}}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def names(R: Optional[dict] = None) -> List[str]:
    return list((R or registry())["models"])


def lookup(name: str, R: Optional[dict] = None) -> Optional[dict]:
    R = R or registry()
    n = _norm(name)
    for k, e in R["models"].items():
        if n == _norm(k) or n in map(_norm, e.get("aliases", [])):
            return dict(e, name=k)
    return None


def expand(spec: str, R: Optional[dict] = None) -> List[str]:
    """'ftq' (a group) / 'ftq13,ftb13' -> entry names; unknown names raise."""
    R = R or registry()
    out = []
    for n in [x for x in spec.split(",") if x.strip()]:
        if _norm(n) in map(_norm, R["groups"]):
            g = next(k for k in R["groups"] if _norm(k) == _norm(n))
            out += R["groups"][g]
            continue
        e = lookup(n, R)
        if e is None:
            raise ValueError(f"unknown GGUF model {n!r}; known: {', '.join(R['models'])}; groups: {', '.join(R['groups'])}")
        out.append(e["name"])
    return list(dict.fromkeys(out))


def model_path(e: dict) -> str:
    return _expand(e["gguf"])


def binary_path(e: dict, R: Optional[dict] = None, device: str = "cpu") -> str:
    """``${LRBENCH_MLIPCPP_BUILDS}/<commit>[-cuda]/bin/mlipcpp-eval``: the CUDA build of the same commit for
    ``device='cuda'`` when there is one (a CPU build cannot run on the GPU; the commit is what must match)."""
    R = R or registry()
    b = R["builds"].get(e["build"], {})
    base = "${LRBENCH_MLIPCPP_BUILDS}/" + e["build"]
    if device == "cuda":
        p = _expand(b.get("binary_cuda", base + "-cuda/bin/mlipcpp-eval"))
        if os.path.isfile(p):
            return p
    return _expand(b.get("binary", base + "/bin/mlipcpp-eval"))


def lib_path(e: dict, R: Optional[dict] = None, device: str = "cpu") -> str:
    """``libmlipcpp_c.so`` of the entry's build (the C API: ``mlipcpp-capi:`` specs, MD in ``lrbench priors``), from
    ``${LRBENCH_MLIPCPP_BUILDS}/<commit>[-cuda]-capi`` (``-DMLIPCPP_BUILD_LAMMPS=ON``) or the build's own dir."""
    R = R or registry()
    b = R["builds"].get(e["build"], {})
    if b.get("lib"):
        return _expand(b["lib"])
    base = _expand("${LRBENCH_MLIPCPP_BUILDS}/" + e["build"])
    dirs = ([base + "-cuda-capi", base + "-cuda"] if device == "cuda" else []) + [base + "-capi", base]
    for d in dirs:
        for sub in ("lib", "src", "bin", "."):
            p = os.path.join(d, sub, "libmlipcpp_c.so")
            if os.path.isfile(p):
                return p
    return os.path.join(dirs[0], "lib", "libmlipcpp_c.so")


def resolve(name: str) -> Optional[str]:
    """Model path for a registered name, else None (lrbench.models.resolve falls through to its own names)."""
    e = lookup(name)
    if e is None:
        return None
    p = model_path(e)
    if "$" in p:
        raise SystemExit(f"{e['name']}: set the environment variables in {e['gguf']!r} (see lrbench.gguf_registry)")
    return p


def entry_for_path(path: str, R: Optional[dict] = None) -> Optional[dict]:
    """The entry whose model file is ``path`` (same file on disk), if any."""
    R = R or registry()
    try:
        rp = os.path.realpath(path)
    except OSError:
        return None
    for k, e in R["models"].items():
        mp = model_path(e)
        if "$" not in mp and os.path.realpath(mp) == rp:
            return dict(e, name=k)
    return None


@lru_cache(maxsize=256)
def _sha256(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256(path: str) -> str:
    st = os.stat(path)
    return _sha256(os.path.realpath(path), st.st_size, st.st_mtime)


def companions(gguf: str) -> Dict[str, str]:
    """{file name as stored in the metadata: absolute path} for every companion the model names."""
    from .backends import gguf_metadata
    meta = gguf_metadata(gguf)
    d = os.path.dirname(os.path.abspath(gguf))
    return {str(meta[k]): os.path.join(d, str(meta[k])) for k in COMPANION_KEYS if meta.get(k)}


def build_commit(binary: str) -> Optional[str]:
    """First token of the ``COMMIT`` file next to the binary (or one directory up), if there is one."""
    for d in (os.path.dirname(binary), os.path.dirname(os.path.dirname(binary))):
        f = os.path.join(d, "COMMIT")
        if os.path.isfile(f):
            t = open(f).read().split()
            return t[0] if t else None
    return None


@lru_cache(maxsize=16)
def _binary_bytes(binary: str, size: int, mtime: float) -> bytes:
    with open(binary, "rb") as f:
        return f.read()


def unread_keys(gguf: str, binary: str) -> List[str]:
    """Metadata keys of the model that do not occur as a string in the binary.

    A heuristic, used *relative* to the build of record: a key the build of record reads but another binary does not
    contain is a key that binary would ignore (e.g. ``lrc_links`` on a build without backbone links). Keys that no
    build contains as a literal (composed names, keys only the trainer reads) are recorded per entry as
    ``keys_not_literal`` by ``add`` and are not held against a binary.
    """
    from .backends import gguf_metadata
    st = os.stat(binary)
    blob = _binary_bytes(os.path.realpath(binary), st.st_size, st.st_mtime)
    return [k for k in gguf_metadata(gguf) if not k.startswith(_NOT_READ) and k.encode() not in blob]


def check_binary(e: dict, binary: str, model: Optional[str] = None) -> dict:
    """Refuse (ValueError) a binary built from another commit than the entry's build, or one that does not know a
    metadata key the build of record knows. ``$LRBENCH_ALLOW_BINARY_MISMATCH=1`` records instead of refusing."""
    model = model or model_path(e)
    have = build_commit(binary)
    want = e["build"]
    problems = []
    if have is None:
        problems.append(f"no COMMIT file next to {binary}: cannot tell which mlip.cpp commit it was built from")
    elif not (have.startswith(want) or want.startswith(have)):
        problems.append(f"binary built from {have}, entry {e['name']} needs {want}")
    unknown = sorted(set(unread_keys(model, binary)) - set(e.get("keys_not_literal", [])))
    if unknown:
        problems.append(f"binary does not know metadata keys the build of record reads: {', '.join(unknown)}")
    out = {"binary": binary, "binary_commit": have, "build": want, "unknown_keys": unknown, "ok": not problems,
           "problems": problems}
    if problems and os.environ.get(ENV_ALLOW) != "1":
        raise ValueError(f"{e['name']}: " + "; ".join(problems) + f" (set {ENV_ALLOW}=1 to score anyway, recorded)")
    return out


def verify_files(e: dict, model: Optional[str] = None, hashes: bool = True) -> dict:
    """Model and companion files exist and (with ``hashes``) match the registered sha256."""
    model = model or model_path(e)
    out = {"model": model, "sha256_ok": None, "companions": {}, "problems": []}
    if not os.path.isfile(model):
        out["problems"].append(f"model file missing: {model}")
        return out
    if hashes and e.get("sha256"):
        out["sha256_ok"] = sha256(model) == e["sha256"]
        if not out["sha256_ok"]:
            out["problems"].append(f"model sha256 differs from the registry ({model})")
    want = e.get("companions", {})
    found = companions(model)
    for name in sorted(set(want) | set(found)):
        p = found.get(name) or os.path.join(os.path.dirname(model), name)
        c = {"path": p, "exists": os.path.isfile(p)}
        if name not in found:
            out["problems"].append(f"registry lists companion {name} but the model does not reference it")
        if not c["exists"]:
            out["problems"].append(f"companion {name} missing next to the model")
        elif hashes and want.get(name):
            c["sha256_ok"] = sha256(p) == want[name]
            if not c["sha256_ok"]:
                out["problems"].append(f"companion {name} sha256 differs from the registry")
        out["companions"][name] = c
    return out


def backend_for(spec: str, binary: Optional[str], device: str = "cpu") -> Optional[dict]:
    """Called by ``make_backend`` for a ``.gguf`` spec: if the file is registered, the binary to use (the entry's own
    build unless one was given) plus the provenance block. None if the file is not registered."""
    e = entry_for_path(spec)
    if e is None:
        return None
    b = binary or binary_path(e, device=device)
    if not os.path.isfile(b):
        raise FileNotFoundError(f"{e['name']}: mlipcpp-eval for build {e['build']} not found at {b}")
    chk = check_binary(e, b, spec)
    files = verify_files(e, spec, hashes=True)
    if files["problems"] and os.environ.get(ENV_ALLOW) != "1":
        raise ValueError(f"{e['name']}: " + "; ".join(files["problems"]))
    prov = {k: e.get(k) for k in ("name", "title", "group", "seed", "build", "level", "released", "licence", "sha256")}
    prov.update(binary_check=chk, files=files)
    return {"binary": b, "provenance": prov, "level": e.get("level")}


def capi_for(spec_path: str, lib: Optional[str], device: str = "cpu") -> Optional[dict]:
    """``backend_for`` for the C API: the entry's ``libmlipcpp_c.so`` unless one was given, checked like a binary."""
    e = entry_for_path(spec_path)
    if e is None:
        return None
    L = lib or lib_path(e, device=device)
    if not os.path.isfile(L):
        raise FileNotFoundError(f"{e['name']}: libmlipcpp_c.so for build {e['build']} not found at {L}")
    chk = check_binary(e, L, spec_path)
    files = verify_files(e, spec_path, hashes=True)
    if files["problems"] and os.environ.get(ENV_ALLOW) != "1":
        raise ValueError(f"{e['name']}: " + "; ".join(files["problems"]))
    prov = {k: e.get(k) for k in ("name", "title", "group", "seed", "build", "level", "released", "licence", "sha256")}
    prov.update(binary_check=chk, files=files)
    return {"lib": L, "provenance": prov}


def entry_from_file(name: str, gguf: str, build: str, binary: str, **fields) -> dict:
    """A registry entry for ``gguf`` evaluated with ``binary`` (the build of record): hashes, companions, and the
    metadata keys this build does not contain (``keys_not_literal``)."""
    comp = {n: sha256(p) for n, p in companions(gguf).items() if os.path.isfile(p)}
    root = os.environ.get("LRBENCH_GGUF_ROOT")
    rel = gguf
    if root and os.path.abspath(gguf).startswith(os.path.abspath(root) + os.sep):
        rel = "${LRBENCH_GGUF_ROOT}/" + os.path.relpath(os.path.abspath(gguf), os.path.abspath(root))
    e = {"gguf": rel, "sha256": sha256(gguf), "companions": comp, "build": build,
         "keys_not_literal": sorted(unread_keys(gguf, binary))}
    e.update(fields)
    return e


# ------------------------------------------------------------------------------------------------------- command line
def _smoke_structure():
    """One small charged, open-shell-free structure: Na+ ... water at 5 A (14 electrons + 10, singlet)."""
    import numpy as np
    from .data import Structure
    pos = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0], [5.0, 0.0, 0.0]])
    return Structure("smoke", ["O", "H", "H", "Na"], pos, 1, 1)


def _cmd_list(a):
    R = registry(a.registry)
    for k, e in R["models"].items():
        print(f"{k:14s} build={e['build']:8s} group={e.get('group', ''):8s} {e.get('title', '')}")


def _cmd_check(a):
    R = registry(a.registry)
    bad = 0
    for n in expand(a.models, R) if a.models else names(R):
        e = dict(R["models"][n], name=n)
        f = verify_files(e, hashes=not a.no_hash)
        b = binary_path(e, R)
        try:
            c = check_binary(e, b) if os.path.isfile(b) else {"ok": False, "problems": [f"binary missing: {b}"]}
        except ValueError as err:
            c = {"ok": False, "problems": [str(err)]}
        probs = f["problems"] + c["problems"]
        bad += bool(probs)
        print(f"{n:14s} {'OK ' if not probs else 'BAD'} " + ("; ".join(probs) if probs else f"build {e['build']}"))
    sys.exit(1 if bad else 0)


def _cmd_smoke(a):
    """Load each model with its own build and evaluate one structure (energy + forces)."""
    import numpy as np
    from .backends import make_backend
    from .cli import read_structures
    R = registry(a.registry)
    s = read_structures(a.xyz)[a.frame] if a.xyz else _smoke_structure()
    rows = {}
    for n in expand(a.models, R) if a.models else names(R):
        e = dict(R["models"][n], name=n)
        try:
            be = make_backend(model_path(e), device=a.device)
            r = be.predict([s])[0]
            f = r.get("forces")
            rows[n] = {"energy": r.get("energy"), "fmax": None if f is None else float(np.abs(f).max()),
                       "finite": r.get("energy") is not None and bool(np.isfinite(r["energy"]))
                       and f is not None and bool(np.isfinite(f).all()),
                       "error": r.get("error"), "binary_commit": be.info().get("gguf_registry", {})
                       .get("binary_check", {}).get("binary_commit")}
        except Exception as err:
            rows[n] = {"energy": None, "error": f"{type(err).__name__}: {err}", "finite": False}
        r = rows[n]
        print(f"{n:14s} {'ok ' if r['finite'] else 'ERR'} E={r['energy']} fmax={r.get('fmax')} "
              f"commit={r.get('binary_commit')} {r.get('error') or ''}", flush=True)
    if a.out:
        json.dump({"structure": s.key, "charge": s.charge, "spin": s.spin, "n_atoms": len(s.species),
                   "results": rows}, open(a.out, "w"), indent=1)
    sys.exit(0 if all(r["finite"] for r in rows.values()) else 1)


def _cmd_add(a):
    e = entry_from_file(a.name, a.gguf, a.build, a.binary, **json.loads(a.fields or "{}"))
    print(json.dumps({a.name: e}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lrbench.gguf_registry", description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", help=f"registry JSON (default ${ENV_REGISTRY}, else configs/gguf_models.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=_cmd_list)
    p = sub.add_parser("check", help="hashes, companions, binary commit and metadata keys of every entry")
    p.add_argument("models", nargs="?", default="")
    p.add_argument("--no-hash", action="store_true")
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("smoke", help="one structure per model, each with its own build")
    p.add_argument("models", nargs="?", default="")
    p.add_argument("--xyz", help="extxyz to take the structure from (default: Na+ ... water at 5 A, Q = +1)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_smoke)
    p = sub.add_parser("add", help="print a registry entry for a model file and its build of record")
    p.add_argument("name")
    p.add_argument("gguf")
    p.add_argument("--build", required=True, help="mlip.cpp commit the model was trained with")
    p.add_argument("--binary", required=True, help="mlipcpp-eval built from that commit")
    p.add_argument("--fields", help="extra JSON fields (title, group, seed, level, released, licence)")
    p.set_defaults(func=_cmd_add)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
