#!/usr/bin/env python3
"""Score externally computed predictions on an lrbench set (same scorer and result schema as `lrbench eval`).
Used for SO3LR(-T) on omol25_val_lr: the `so3lr eval` CLI pads structures into a few size buckets (one JAX compilation
per bucket), whereas lrbench's ASE path compiles once per distinct atom count.

    python score_frames.py export SET DATA_DIR OUT.extxyz          # records -> extxyz (info id, charge, multiplicity)
    python score_frames.py score SET DATA_DIR PRED.extxyz OUT.json --label L --model-level LEVEL [--energy-key energy_so3lr]
                                [--forces-key forces_so3lr] [--n-boot 1000]
PRED.extxyz must keep the export order (so3lr eval does). Element offsets are fitted in sample as in `lrbench eval`;
a model level different from the labels is recorded as an allowed mismatch."""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")


def export(name, data, out):
    from ase import Atoms
    from ase.io import write
    from lrbench.data import load_set, needed_terms, term_structure
    ds = load_set(name, data)
    F = []
    for t in needed_terms(ds):
        s = term_structure(ds, t)
        a = Atoms(s.species, positions=s.positions)
        a.info.update(id=t, charge=int(s.charge), multiplicity=int(s.spin))
        F.append(a)
    write(out, F, format="extxyz")
    print(out, len(F))


def score(name, data, pred, out, *a):
    from ase.io import read
    from lrbench.cli import _jsonable, apply_offsets, element_offsets
    from lrbench.data import load_set, needed_terms
    from lrbench.metrics import Preds, SetScorer
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    ek, fk = opt("--energy-key", "energy_so3lr"), opt("--forces-key", "forces_so3lr")
    ds = load_set(name, data); terms = needed_terms(ds); P = read(pred, ":")
    assert len(P) == len(terms), (len(P), len(terms))
    m = {}
    for t, p in zip(terms, P):
        e = p.info.get(ek)
        e = None if e is None or not np.isfinite(float(np.ravel(e)[0])) else float(np.ravel(e)[0])
        f = p.arrays.get(fk)
        m[t] = {"energy": e, "forces": None if f is None else np.asarray(f, float), "charges": None,
                "error": None if e is not None else "no prediction"}
    data_ = {name: ds}; preds = [{name: m}]
    off = element_offsets(data_, preds, "fit")
    if off:
        apply_offsets(data_, preds, off)
    lv = opt("--model-level", "unknown")
    R = {"label": opt("--label", pred), "level": {"model_level": lv, "mismatch": {name: ds.levels()} if lv not in ds.levels() else {}},
         "members": [{"spec": f"external predictions: {pred}"}], "element_offsets": off,
         "notes": [f"scored from external predictions ({pred}) with lrbench.metrics.SetScorer"] +
                  ([f"LEVEL MISMATCH (allowed): model {lv} vs labels {ds.levels()}"] if lv not in ds.levels() else []),
         "sets": {name: SetScorer(ds, Preds([preds[0][name]]), n_boot=int(opt("--n-boot", 1000))).run()},
         "data": {name: ds.source}}
    R["sets"][name]["predictions"] = {t: [preds[0][name][t]["energy"]] for t in terms}
    json.dump(_jsonable(R), open(out, "w"))
    print(out)


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]](*sys.argv[2:])
