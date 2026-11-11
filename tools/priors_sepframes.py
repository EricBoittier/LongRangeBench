#!/usr/bin/env python3
"""Separated / single-component frame probes for `lrbench priors --sep-frames` (the X9 offset-by-Q population).

    python priors_sepframes.py VAL_SEL.npz SEL.json ATOM_REFS.json OUT.extxyz [--clean KEEP.npy] [--n-single 15]

separated: the frames of SEL.json (val_sel_clean, multi-component, Q > 0, single-link gap >= 12 A; written by the
sep_cation diagnosis, results_conv/sep_cation/work/sel.json). single: up to --n-single single-component
(ncomp == 1) val_sel_clean frames per charge Q present in the separated set, picked with a fixed seed, as the
same-Q control. Each frame: info charge, spin_multiplicity, E_dft (eV), ref_offset = sum of the OMol25 neutral-atom
element references (ATOM_REFS 'composition'; the Q-independent refit of X9, so dE = E_model + ref_offset - E_DFT for
models with reference-subtracted energies such as PET-OMol-S and its fine-tunes), stratum, val_sel index, gap."""
import argparse
import json

import numpy as np
from ase import Atoms
from ase.io import write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"), ap.add_argument("sel"), ap.add_argument("refs"), ap.add_argument("out")
    ap.add_argument("--clean")
    ap.add_argument("--n-single", type=int, default=15)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    off = np.concatenate([[0], np.cumsum(z["n"])])
    comp = json.load(open(a.refs))["composition"]
    S = json.load(open(a.sel))

    def frame(i, stratum, gap=None):
        Z = z["Z"][off[i]:off[i + 1]]
        at = Atoms(numbers=Z, positions=z["R"][off[i]:off[i + 1]])
        at.info = {"charge": int(z["charge"][i]), "spin_multiplicity": int(z["spin_multiplicity"][i]),
                   "E_dft": float(z["E"][i]), "ref_offset": float(sum(comp[str(int(x))] for x in Z)),
                   "stratum": stratum, "val_sel_idx": int(i), "domain": str(z["domain"][i])}
        if gap is not None:
            at.info["gap"] = float(gap)
        return at

    frames = [frame(s["i"], "separated", s["gap"]) for s in S]
    clean = np.zeros(len(z["charge"]), bool)
    clean[np.load(a.clean) if a.clean else np.arange(len(clean))] = True
    rng = np.random.default_rng(0)
    for q in sorted({s["Q"] for s in S}):
        idx = np.where(clean & (z["ncomp"] == 1) & (z["charge"] == q))[0]
        for i in rng.choice(idx, min(a.n_single, len(idx)), replace=False):
            frames.append(frame(int(i), "single"))
    write(a.out, frames, format="extxyz")
    print(a.out, sum(f.info["stratum"] == "separated" for f in frames), "separated,",
          sum(f.info["stratum"] == "single" for f in frames), "single")


if __name__ == "__main__":
    main()
