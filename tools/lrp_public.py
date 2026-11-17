#!/usr/bin/env python3
"""Make the notebook data bundle public-safe, and check that it is.

    python tools/lrp_public.py --check notebooks/data/lr_program          # gate only (exit 1 on any hit)
    python tools/lrp_public.py --apply notebooks/data/lr_program          # rewrite labels in place, then gate

The program's score files are named by internal run labels. The public bundle names every scored file by arm, seed
and weights only (``P_s14_ema``, ``AL_s13_raw``) and names hardware by GPU type. The mapping from internal labels to
public ones, and the list of strings that must never reach the bundle, live in a private JSON file outside the
repository (``LRP_PUBLIC_MAP``); this module holds only the mechanics:

* JSON files are rewritten as text (every string, key and value).
* ``chemiscope.json`` holds gzip+base64 blobs: each blob is decoded, rewritten and re-encoded.
* ``.npz`` files: array names and string arrays are rewritten; numeric arrays are untouched.

The gate runs on the same decoded views, so compressed or encoded content cannot hide a hit.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import pathlib
import re

import numpy as np

B64 = re.compile(r"[A-Za-z0-9+/=]{400,}")
DEFAULT_MAP = pathlib.Path(os.environ.get("LRNB_DATA_ROOT", "~/data")).expanduser() / "lr-program" / "notebook_publish" / "public_map.json"


def load_map(path=None):
    p = pathlib.Path(path or os.environ.get("LRP_PUBLIC_MAP") or DEFAULT_MAP)
    if not p.exists():
        raise SystemExit(f"public map not found at {p}; set LRP_PUBLIC_MAP (the file is private and not in the repo)")
    m = json.loads(p.read_text())
    m["_rx"] = [(re.compile(a), b) for a, b in m.get("replace", [])]
    m["_bad"] = [re.compile(a) for a in m.get("forbidden", [])]
    return m


def rewrite(text: str, M) -> str:
    for rx, rep in M["_rx"]:
        text = rx.sub(rep, text)
    return text


def hits(text: str, M):
    out = []
    for rx in M["_bad"]:
        for m in rx.finditer(text):
            a, b = max(0, m.start() - 40), min(len(text), m.end() + 40)
            out.append((rx.pattern, text[a:b].replace("\n", " ")))
    return out


def _blob_decode(b: str) -> str:
    return gzip.decompress(base64.b64decode(b)).decode()


def _blob_encode(t: str) -> str:
    return base64.b64encode(gzip.compress(t.encode(), 9)).decode()


def views(path: pathlib.Path):
    """(name, text) views of a bundle file in which every string is visible"""
    if path.suffix == ".json":
        t = path.read_text()
        try:
            d = json.loads(t)
        except Exception:
            yield path.name, t
            return
        if not (isinstance(d, dict) and isinstance(d.get("chemiscope"), dict)):
            yield path.name, B64.sub("<base64>", t)   # embedded PNGs: random letters, nothing to read
        else:   # the raw text is base64 (random letters): scan the decoded blobs and everything else instead
            yield path.name, json.dumps({k: v for k, v in d.items() if k != "chemiscope"}) + "\n".join(d["chemiscope"])
            for k, b in d["chemiscope"].items():
                if isinstance(b, str):
                    yield f"{path.name}:{k}", _blob_decode(b)
    elif path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        yield f"{path.name}:keys", "\n".join(z.files)
        for k in z.files:
            a = z[k]
            if a.dtype.kind in "US":
                yield f"{path.name}:{k}", "\n".join(map(str, a.ravel().tolist()))
    elif path.suffix in (".txt", ".md", ".csv", ".py", ".sh", ".html", ".js"):
        yield path.name, path.read_text(errors="replace")


def apply_file(path: pathlib.Path, M) -> None:
    if path.suffix == ".json":
        t = path.read_text()
        try:
            d = json.loads(t)
        except Exception:
            d = None
        if isinstance(d, dict) and isinstance(d.get("chemiscope"), dict):
            d["chemiscope"] = {rewrite(k, M): _blob_encode(rewrite(_blob_decode(b), M)) for k, b in d["chemiscope"].items()}
            d["views"] = json.loads(rewrite(json.dumps(d.get("views", [])), M))
            path.write_text(json.dumps(d, separators=(",", ":")))
        else:
            path.write_text(rewrite(t, M))
    elif path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        out = {}
        for k in z.files:
            a = z[k]
            if a.dtype.kind in "US":
                a = np.array([rewrite(str(x), M) for x in a.ravel().tolist()]).reshape(a.shape)
            nk = rewrite(k, M)
            assert nk not in out, f"label collision after rewrite: {k} -> {nk}"
            out[nk] = a
        buf = io.BytesIO()
        np.savez_compressed(buf, **out)
        path.write_bytes(buf.getvalue())


def check_paths(paths, M, verbose=True):
    n = 0
    for p in paths:
        p = pathlib.Path(p)
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in files:
            for name, t in views(f):
                for pat, ctx in hits(t, M):
                    n += 1
                    if verbose and n <= 60:
                        print(f"[gate] {name}: /{pat}/ ... {ctx}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--apply", action="store_true", help="rewrite labels in place before the gate")
    ap.add_argument("--check", action="store_true", help="gate only")
    ap.add_argument("--map", default=None)
    a = ap.parse_args(argv)
    M = load_map(a.map)
    if a.apply:
        for p in a.paths:
            p = pathlib.Path(p)
            for f in sorted(x for x in (p.rglob("*") if p.is_dir() else [p]) if x.is_file()):
                if f.suffix in (".json", ".npz"):
                    apply_file(f, M)
    n = check_paths(a.paths, M)
    print(f"[gate] {n} hit(s) in {', '.join(a.paths)}")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
