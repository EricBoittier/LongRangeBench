#!/usr/bin/env python3
"""LOREM (lorem-jax, lorem-tmlr-archive SN2 checkpoints) on the lrbench scans: E_int and the summed scalar latent
charge on the neutral CH3F fragment along the SN2 scan and the size-consistency scan, and the box scan
(E(periodic box, L) - E(open)) for neutral CH3F and charged CH3F...I-.

    python lorem_scan.py CHECKPOINT_DIR SCANS_DIR OUT.json [--label NAME]

Non-PBC convention: the shipped non-PBC checkpoints were trained with V_i = sum_j q_j / r_ij, today's jax-pme halves
the non-PBC potential (lorem-tmlr-archive README). Every open-boundary evaluation here multiplies the non-PBC Ewald
potential by 2 (the README's fix, validated in etc/analysis notebook 13). For periodic structures two conventions are
reported: 'pbc_x1' (today's PBC Ewald, untouched, which the README calls unchanged) and 'pbc_x2' (the same factor 2,
i.e. the convention the model was trained with if the pre-release code used one convention for both). The latent
charge is the scalar channel fed to Ewald (MLP [2d, 1] output), unconstrained (LOREM has no total-charge input)."""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)


def load(ckpt):
    import lorem.models.mlip as M
    from lorem.calculator import Calculator
    calc = Calculator.from_checkpoint(ckpt)
    base = M.Ewald
    state = {"f_open": 2.0, "f_pbc": 1.0}

    def Ewald2(*a, **k):  # jaxpme Ewald() is a factory; only .potentials is used by lorem.models.mlip
        from types import SimpleNamespace
        c = base(*a, **k)

        def potentials(q, sr, nopbc, pbc):
            per = sr.pbc_mask[sr.atom_to_structure]
            f = jnp.where(per, state["f_pbc"], state["f_open"])
            return c.potentials(q, sr, nopbc, pbc) * f
        return SimpleNamespace(potentials=potentials)

    M.Ewald = Ewald2
    return calc, state


def run(calc, atoms):
    """energy (eV, with elemental baseline), forces, scalar latent charges; no jit (the factor is a Python global)."""
    calc.setup(atoms)
    b = calc.batch
    mdl = _MODEL["m"]
    is_q = lambda m, name: name == "__call__" and isinstance(getattr(m, "features", None), (list, tuple)) and len(m.features) == 2

    def efn(pos):
        sr = b.sr._replace(positions=pos)
        e, inter = mdl.apply(calc.params, b.atomic_numbers, sr, b.nopbc, b.pbc, capture_intermediates=is_q,
                             mutable=["intermediates"])
        return jnp.sum(e * sr.atom_mask), inter

    (e, inter), g = jax.value_and_grad(efn, has_aux=True)(b.sr.positions)
    n = len(atoms)
    qs = [np.asarray(v[0]) for v in jax.tree_util.tree_leaves(inter["intermediates"], is_leaf=lambda x: isinstance(x, tuple))]
    q = np.asarray(qs[-1]).reshape(-1)[:n] if qs else None
    E = float(e) + sum(calc.species_weights[int(z)] for z in atoms.numbers)
    return E, -np.asarray(g)[:n], q


_MODEL = {}


def main(ckpt, scans, out, label="lorem"):
    from ase.io import read
    from pathlib import Path
    from marathon.io import from_dict as fd, read_yaml
    calc, state = load(ckpt)
    _MODEL["m"] = fd(read_yaml(Path(ckpt) / "model/model.yaml"))
    res = {"label": label, "checkpoint": ckpt, "definition": __doc__}
    # SN2 scan (I- pulled along C->I): frames alternate AB, A(CH3F); last frame is I- alone
    S = read(f"{scans}/sn2_scan.extxyz", ":")
    eB = run(calc, S[-1])[0]
    rows = []
    F_far = None
    for i in range(0, len(S) - 1, 2):
        eAB, fAB, q = run(calc, S[i]); eA = run(calc, S[i + 1])[0]
        rows.append(dict(d=float(S[i].info["d"]), E_int_meV=(eAB - eA - eB) * 1e3, q_CH3F=float(q[:5].sum()) if q is not None else None,
                         q_I=float(q[5]) if q is not None else None, q_total=float(q.sum()) if q is not None else None, F=fAB[:5]))
    F_far = rows[-1]["F"]
    for r in rows:
        r["max_dF_CH3F"] = float(np.linalg.norm(r.pop("F") - F_far, axis=1).max())
    res["sn2_scan"] = rows
    # size consistency: CH3F...I- only (the other systems have elements outside the SN2 training set)
    Z = read(f"{scans}/sizecons.extxyz", ":")
    by = {}
    for a in Z:
        if a.info["sys"] == "CH3F_I":
            e, _, q = run(calc, a)
            by.setdefault(a.info["d"], {})[a.info["role"]] = (e, q)
    res["sizecons"] = [dict(sys="CH3F_I", d=float(d), E_int_meV=(v["AB"][0] - v["A"][0] - v["B"][0]) * 1e3,
                            q_A_latent=None if v["AB"][1] is None else float(v["AB"][1][:5].sum()),
                            q_B_latent=None if v["AB"][1] is None else float(v["AB"][1][5]))
                       for d, v in sorted(by.items())]
    # box scan, both periodic conventions
    B = read(f"{scans}/box_scan.extxyz", ":")
    res["box_scan"] = {}
    for conv, f in (("pbc_x1", 1.0), ("pbc_x2", 2.0)):
        state["f_pbc"] = f
        rows = []
        for sy in ("CH3F", "CH3F_I"):
            fr = [a for a in B if a.info["sys"] == sy]
            e0, _, q0 = run(calc, next(a for a in fr if a.info["role"] == "open"))
            for a in fr:
                if a.info["role"] == "box":
                    e, _, q = run(calc, a)
                    rows.append(dict(sys=sy, L=float(a.info["L"]), dE_box_minus_open_meV=(e - e0) * 1e3,
                                     q_sum=None if q is None else float(q.sum()),
                                     q_sum_open=None if q0 is None else float(q0.sum())))
        fit = {}
        for sy in ("CH3F", "CH3F_I"):
            s = [r for r in rows if r["sys"] == sy and abs(r["dE_box_minus_open_meV"]) > 1e-3]
            if len(s) >= 3:
                fit[sy] = float(np.polyfit(np.log([r["L"] for r in s]), np.log([abs(r["dE_box_minus_open_meV"]) for r in s]), 1)[0])
        res["box_scan"][conv] = {"rows": rows, "power_law_exponent": fit}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"sn2_far": res["sn2_scan"][-1], "sizecons": res["sizecons"][-3:], "box": {k: v["power_law_exponent"] for k, v in res["box_scan"].items()}}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    lab = a[a.index("--label") + 1] if "--label" in a else "lorem"
    main(a[0], a[1], a[2], lab)
