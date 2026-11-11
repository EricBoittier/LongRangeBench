"""A tiny self-contained set in the canonical schema whose "DFT" references are a Lennard-Jones calculator's own
values. Scoring that same calculator must give zero error, tail fraction 1 and ddE slope 1; a truncated LJ must give
tail fraction 0. It needs no download and no GPU, so it is what the quickstart, the tests and CI run on.

    lrbench toy ./toy-data        # writes canonical/toy.jsonl + canonical/toy.groups.json
    lrbench eval --model "ase:ase.calculators.lj:LennardJones?sigma=2.5&epsilon=0.05&rc=40.0" \
        --sets toy --data ./toy-data --out toy.json

Contents: 2 dimer scans (ion_pair / same_sign classes, 7 distances each, with fragment slices and an E_mp prior
beyond 8 A), one record flagged as a DFT pathology (never scored), and 3 fixed-charge ddE quartets in the
``wire_disc_near_vs_far`` series."""
from __future__ import annotations

import json
import os

import numpy as np

LJ_SPEC = "ase:ase.calculators.lj:LennardJones?sigma=2.5&epsilon=0.05&rc=40.0"
LEVEL = "wB97M-V/def2-TZVPD"


def lj_calc(rc: float = 40.0):
    from ase.calculators.lj import LennardJones
    return LennardJones(sigma=2.5, epsilon=0.05, rc=rc)


def _e(sp, x):
    from ase import Atoms
    a = Atoms(sp, positions=x)
    a.calc = lj_calc()
    return float(a.get_potential_energy()), a.get_forces().tolist()


def _rec(rid, sp, x, fid=None, fq=None, **kw):
    e, f = _e(sp, x)
    r = dict(id=rid, species=sp, positions=np.asarray(x).tolist(), cell=None, pbc=[False] * 3, charge=0, spin=1,
             energy=e, forces=f, formal_charges=[0] * len(sp), fragment_id=fid, fragment_charges=fq,
             fragment_spins=None if fq is None else [1] * len(fq), flags=[], refs={}, extra={}, level=LEVEL)
    r.update(kw)
    return r


def records():
    """(records, groups) of the toy set."""
    A = np.array([[0, 0, 0], [1.1, 0, 0]])
    B = np.array([[0, 0, 0], [0, 1.2, 0], [0, 0, 1.3]])
    recs, groups = [], []
    for o, cls in enumerate(["ion_pair", "same_sign"]):
        for d0 in [3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
            rid = f"AB:o{o}:d{d0}"
            x = np.vstack([A, B + [d0, 0.3 * o, 0]])
            sp = ["Ar", "Ar", "Ne", "Ne", "Ne"]
            recs.append(_rec(rid, sp, x, fid=[0, 0, 1, 1, 1], fq=[0, 0], cls=cls, group=f"AB:o{o}", role="AB",
                             distance=d0))
            ea, _ = _e(sp[:2], x[:2])
            eb, _ = _e(sp[2:], x[2:])
            if d0 >= 8:  # the "physical prior" of the toy: the LJ interaction itself (stands in for E_mp)
                recs[-1]["refs"]["E_mp"] = recs[-1]["energy"] - ea - eb
            groups.append({"name": f"E_int:{rid}", "kind": "E_int", "ref": recs[-1]["energy"] - ea - eb,
                           "terms": [[rid, 1.0], [f"{rid}#frag=0", -1.0], [f"{rid}#frag=1", -1.0]],
                           "meta": {"cls": cls, "dmin": d0, "pair": "AB"}})
    bad = _rec("bad", ["Ar", "Ar"], A * 1.5, cls="ion_pair", distance=None, flags=["dft_pathology_do_not_score"])
    bad["energy"] = 1e6
    recs.append(bad)
    groups.append({"name": "E_int:bad", "kind": "E_int", "ref": 1e6, "terms": [["bad", 1.0]],
                   "meta": {"flags": ["dft_pathology_do_not_score"]}})
    for k, s in enumerate([1.0, 1.5, 2.0]):
        ids = []
        for near in (4.0, 30.0):
            for site in (0, 1):
                rid = f"w{k}:n{near}:s{site}"
                x = np.array([[0, 0, 0], [3.0 + 0.3 * site, 0, 0], [near, s, 0]])
                recs.append(_rec(rid, ["Ar", "Ar", "Kr"], x, cls="wire", group=f"w{k}:n{near}", role=f"s{site}"))
                ids.append(rid)
        e = {r["id"]: r["energy"] for r in recs}
        groups.append({"name": f"ddE:w{k}", "kind": "ddE",
                       "terms": [[ids[1], 1], [ids[0], -1], [ids[3], -1], [ids[2], 1]],
                       "ref": e[ids[1]] - e[ids[0]] - e[ids[3]] + e[ids[2]],
                       "meta": {"series": "wire_disc_near_vs_far"}})
    return recs, groups


def build(outdir: str, name: str = "toy") -> str:
    """Write ``<outdir>/canonical/<name>.jsonl`` and ``.groups.json``; returns ``<outdir>`` (pass it to ``--data``)."""
    d = os.path.join(outdir, "canonical")
    os.makedirs(d, exist_ok=True)
    recs, groups = records()
    with open(os.path.join(d, f"{name}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, f"{name}.groups.json"), "w") as f:
        json.dump(groups, f)
    return outdir
