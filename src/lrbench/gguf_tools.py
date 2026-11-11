"""GGUF weight averaging, only when it is trivial: identical tensor names, shapes and dtypes (F32/F16 only) and the same
metadata. The averaged file is a byte copy of the first member with every tensor's data region overwritten in place,
so metadata and layout are untouched.

Caveat: averaging weights of independently trained seeds is not a meaningful model unless they share a basin
(same init and short divergence); lrbench reports it as a separate row ("weight average") and always scores the
prediction-averaged ensemble as the ensemble result."""
from __future__ import annotations

import shutil
from typing import List, Tuple

import numpy as np


def compatible(paths: List[str]) -> Tuple[bool, str]:
    try:
        from gguf import GGUFReader
    except ImportError:
        return False, "the gguf python package is not installed"
    R = [GGUFReader(p) for p in paths]
    sig = lambda r: [(t.name, tuple(int(x) for x in t.shape), t.tensor_type.name) for t in r.tensors]
    s0 = sig(R[0])
    for p, r in zip(paths[1:], R[1:]):
        if sig(r) != s0:
            return False, f"tensor names/shapes/dtypes differ between {paths[0]} and {p}"
    bad = {d for _, _, d in s0} - {"F32", "F16"}
    if bad:
        return False, f"unsupported tensor dtypes for averaging: {sorted(bad)}"

    def meta(r):
        out = {}
        for k, f in r.fields.items():
            if k.startswith("GGUF."):
                continue
            try:
                out[k] = f.contents()
            except Exception:
                out[k] = None
        return out
    m0 = meta(R[0])
    for p, r in zip(paths[1:], R[1:]):
        diff = [k for k in set(m0) | set(meta(r)) if m0.get(k) != meta(r).get(k)]
        if diff:
            return False, f"metadata differs ({', '.join(sorted(diff)[:5])}) between {paths[0]} and {p}"
    return True, "ok"


def average(paths: List[str], out: str) -> Tuple[bool, str]:
    ok, why = compatible(paths)
    if not ok:
        return False, why
    from gguf import GGUFReader
    R = [GGUFReader(p) for p in paths]
    shutil.copyfile(paths[0], out)
    mm = np.memmap(out, mode="r+", dtype=np.uint8)
    for k, t in enumerate(R[0].tensors):
        dt = np.float32 if t.tensor_type.name == "F32" else np.float16
        acc = np.mean([np.asarray(r.tensors[k].data, dtype=np.float64) for r in R], axis=0).astype(dt)
        b = acc.tobytes()
        assert len(b) == t.n_bytes, (t.name, len(b), t.n_bytes)
        mm[t.data_offset:t.data_offset + t.n_bytes] = np.frombuffer(b, np.uint8)
    mm.flush()
    del mm
    return True, f"averaged {len(paths)} GGUFs ({len(R[0].tensors)} tensors) into {out}"


def les_gate(path: str):
    """LES amplitude g of a Plato-PET-LR GGUF (tensor plato_les_gate); None if absent or unreadable."""
    if not str(path).endswith(".gguf"):
        return None
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    t = next((t for t in GGUFReader(path).tensors if t.name == "plato_les_gate"), None)
    return None if t is None else float(np.asarray(t.data, dtype=np.float64).ravel()[0])
