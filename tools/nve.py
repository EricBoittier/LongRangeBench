#!/usr/bin/env python3
"""NVE stability of one Na+ in the lrbench `water3k` box (1,000 H2O, 31.07 A, periodic; the water molecule closest to
the box centre is replaced by Na+, total charge +1, neutralising background implied by each model's periodic
treatment), ASE VelocityVerlet at 0.5 fs from the box's stored momenta (Na+ starts at rest; no re-equilibration, so
the first ps carries the insertion transient).

    python nve.py MODEL_SPEC WATER3K.extxyz STEPS OUT.json [--device cuda] [--mlipcpp-lib LIB] [--every 10]

Reports: ms/step (median after 5 warm-up steps) and the projection to 100 ps (200,000 steps); total-energy drift
(least-squares slope, meV/atom/ps, and over the last half) and RMS fluctuation (meV/atom); temperature; Na-O RDF
(minimum image, frames every 20 steps after the first 200) and its first peak / first minimum / coordination number.
fp32 for mlip.cpp unless the model says otherwise; NVIDIA_TF32_OVERRIDE is removed."""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")


def build(path):
    from ase.io import read
    a = read(path)
    O = np.where(a.numbers == 8)[0]
    c = a.cell.lengths() / 2
    k = O[np.argmin(np.linalg.norm(a.positions[O] - c, axis=1))]
    H = [i for i in np.where(a.numbers == 1)[0] if np.linalg.norm(a.get_distance(k, i, mic=True)) < 1.25][:2]
    p = a.get_momenta() if "momenta" in a.arrays else np.zeros((len(a), 3))
    a.numbers[k] = 11
    p[k] = 0
    del a[H]
    a.set_momenta(np.delete(p, H, axis=0))
    a.info.update(charge=1, spin_multiplicity=1, spin=1)
    return a, int(np.where(a.numbers == 11)[0][0])


def rdf(frames, ion, cell, rmax=8.0, nb=160):
    L = np.diag(cell); edges = np.linspace(0, rmax, nb + 1); h = np.zeros(nb); nO = 0
    for x, Z in frames:
        d = x[Z == 8] - x[ion]; d -= L * np.round(d / L); r = np.linalg.norm(d, axis=1)
        h += np.histogram(r, edges)[0]; nO = (Z == 8).sum()
    rho = nO / np.prod(L); shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = h / (len(frames) * rho * shell); rc = 0.5 * (edges[1:] + edges[:-1])
    i1 = int(np.argmax(g)); j = i1 + int(np.argmin(g[i1:i1 + 40]))
    cn = float(np.sum(h[:j + 1]) / len(frames))
    return dict(r=rc.round(3).tolist(), g=g.round(4).tolist(), peak_r=float(rc[i1]), peak_g=float(g[i1]),
                min_r=float(rc[j]), coordination=cn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("box"); ap.add_argument("steps", type=int); ap.add_argument("out")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--mlipcpp-lib"); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--label")
    a = ap.parse_args()
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from lrbench.backends import make_backend
    at, ion = build(a.box)
    be = make_backend(a.model, device=a.device, mlipcpp_lib=a.mlipcpp_lib)
    at.calc = be.ase_calculator(1, 1) if hasattr(be, "ase_calculator") else be.calc
    md = VelocityVerlet(at, timestep=0.5 * units.fs)
    E, T, ts, frames = [], [], [], []
    n = len(at)
    for s in range(a.steps + 1):
        if s:
            t0 = time.perf_counter(); md.run(1); ts.append(time.perf_counter() - t0)
        if s % a.every == 0:
            E.append((at.get_potential_energy() + at.get_kinetic_energy()) / n * 1e3); T.append(at.get_temperature())
        if s >= 200 and s % 20 == 0:
            frames.append((at.positions.copy(), at.numbers.copy()))
    E = np.array(E); t_ps = np.arange(len(E)) * a.every * 0.5e-3
    half = len(E) // 2
    ms = float(np.median(ts[5:])) * 1e3 if len(ts) > 5 else None
    out = dict(model=a.model, label=a.label or a.model, n_atoms=n, steps=a.steps, dt_fs=0.5, box=a.box,
               ms_per_step=ms, projected_100ps_h=None if ms is None else ms * 200000 / 3.6e6,
               drift_meV_atom_ps=float(np.polyfit(t_ps, E, 1)[0]),
               drift_last_half_meV_atom_ps=float(np.polyfit(t_ps[half:], E[half:], 1)[0]) if half > 2 else None,
               etot_rms_meV_atom=float(np.std(E)), T_mean_K=float(np.mean(T)), T_final_K=float(T[-1]),
               etot_meV_atom=E.round(4).tolist(), t_ps=t_ps.round(4).tolist(),
               rdf_NaO=rdf(frames, ion, at.cell.array) if frames else None, backend=be.info(),
               note="Na+ replaces the water nearest the box centre; starts at rest; no re-equilibration")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("label", "ms_per_step", "projected_100ps_h", "drift_meV_atom_ps", "etot_rms_meV_atom", "T_mean_K")}))


if __name__ == "__main__":
    main()
