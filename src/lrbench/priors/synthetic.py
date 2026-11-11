"""Synthetic analytic models for the priors suite: a clean point-charge model and one planted defect per check.

Every model is an ASE calculator (fp64), so it runs through ``ASEBackend`` exactly like an external model, and can be
named on the command line as ``ase:lrbench.priors.synthetic:PointCharge`` (or any class below, with ``?kw=v``).

The clean model (``PointCharge``):
  - per-atom charges from a fixed element table ``Q0`` (Na/Li/K +1, Cl/F -1, O -0.8, H +0.4, C 0), so an H2O is
    neutral and polar, and a bare ion carries its integer charge; ``charges`` are returned in ``results``;
  - energy = sum_i e_Z + damped Coulomb over all atom pairs, 14.40 q_i q_j erf(r/sigma)/r (exact 1/r beyond ~3 sigma),
    + a charge-state term a x^2 + b x with x = Q - sum q0 (the model reacts to the total charge, convex in N) and a spin
    term c (S - 1). Test systems keep Q = sum q0 for every fragment, so the charge-state term vanishes there;
  - forces: analytic Coulomb gradient (exact, conservative);
  - ``check_state`` is info-aware: a change of info charge / spin_multiplicity invalidates the cached result.

Planted defects (each subclasses the clean model and breaks one prior):
  ``Switched``      plateau: the inter-cluster Coulomb is replaced by a constant over a gap window (default
                    7.5-9.0 A) and ramps back over ``ramp`` A (the FTq seam band); conservative (forces are the gradient)
  ``Leaky``         global charge equilibration: q_i = (1-w) q0_i + w Q/N, so a neutral partner carries charge (Fukui leak)
  ``Caching``       ASE's default check_state (positions/numbers/cell only): a call that differs only in charge/spin
                    returns the previous result (the upet 0.3.1 bug)
  ``Flat``          inter-cluster Coulomb times a smooth cutoff (4.5 -> 5.5 A gap): no tail beyond the cutoff
  ``SpuriousLR``    adds 14.40 * 0.1^2 / R_cc between every pair of clusters, neutral ones included (non-local leak)
  ``SepOffset``     -1 eV per unit charge on every charged cluster when the structure has >= 2 clusters (X9 separated-
                    cation offset)
  ``QBlind``        ignores the total charge and spin (charge-state term off)
  ``NonConservative`` forces scaled by 1.2 (not the gradient of the energy)
  ``Discontinuous`` +0.2 eV energy step at a gap of 7.03 A with no force
  ``NonInvariant``  energy + 0.01 eV/A * x of the first atom (breaks translation / rotation invariance)
  ``Noisy``         energy + N(0, 1e-3) eV per call (non-deterministic)
  ``BoxDependent``  energy + 1 eV A / L when a cell is present (periodic result does not converge to open)
  ``IPShift``       charge-state term shifted by +0.5 eV per removed electron (vertical IP error, the fine-tune damage)
"""
from __future__ import annotations

import math

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

KE = 14.399645  # e^2 / (4 pi eps0) in eV A
Q0 = {"H": 0.4, "O": -0.8, "C": 0.0, "N": 0.0, "Na": 1.0, "Li": 1.0, "K": 1.0, "Cl": -1.0, "F": -1.0, "Br": -1.0}
E0 = {"H": -13.6, "O": -2040.0, "C": -1030.0, "N": -1485.0, "Na": -4415.0, "Li": -203.0, "K": -16323.0,
      "Cl": -12520.0, "F": -2714.0, "Br": -70000.0}
LINK = 3.0  # A, single-linkage cluster cutoff


def clusters(x: np.ndarray, link: float = LINK):
    """Single-linkage clusters at ``link`` A -> (labels, n_clusters)."""
    n = len(x)
    lab = -np.ones(n, int)
    k = 0
    D = np.linalg.norm(x[:, None] - x[None], axis=-1)
    for i in range(n):
        if lab[i] >= 0:
            continue
        st, lab[i] = [i], k
        while st:
            j = st.pop()
            for m in np.where((D[j] < link) & (lab < 0))[0]:
                lab[m] = k
                st.append(m)
        k += 1
    return lab, k


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


class PointCharge(Calculator):
    implemented_properties = ["energy", "forces", "charges"]
    charge_spin = True
    lrbench_meta = {"name": "synthetic point-charge model", "level": "synthetic"}

    def __init__(self, sigma: float = 1.0, a: float = 5.0, b: float = 3.0, c: float = 0.2, **kw):
        super().__init__(**kw)
        self.sigma, self.a, self.b, self.c = sigma, a, b, c

    # ------------------------------------------------------------------ caching
    def check_state(self, atoms, tol=1e-15):
        ch = super().check_state(atoms, tol)
        old = getattr(self, "_info_state", None)
        if old is not None and old != self._state(atoms):
            ch = list(ch) + ["info"]
        return ch

    @staticmethod
    def _state(atoms):
        return (int(atoms.info.get("charge", 0)), int(atoms.info.get("spin_multiplicity", atoms.info.get("spin", 1))))

    # ------------------------------------------------------------------ model
    def q(self, sp, x, Q):
        return np.array([Q0.get(s, 0.0) for s in sp])

    def pair_scale(self, x, lab):
        """Multiplier on each atom pair's Coulomb term (1 = clean) and its gradient wrt positions: (S, dS) with
        S (n, n) and dS (n, n, n, 3) or None when S is constant."""
        return None, None

    def charge_state(self, sp, x, Q, S, lab, ncl):
        x0 = Q - sum(Q0.get(s, 0.0) for s in sp)
        return self.a * x0 * x0 + self.b * x0 + self.c * (S - 1)

    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        return 0.0, np.zeros_like(x)

    def compute(self, sp, x, Q, S, atoms=None):
        n = len(sp)
        lab, ncl = clusters(x)
        q = self.q(sp, x, Q)
        E = sum(E0.get(s, -100.0) for s in sp)
        F = np.zeros((n, 3))
        Sm, dS = self.pair_scale(x, lab)
        for i in range(n):
            for j in range(i + 1, n):
                d = x[i] - x[j]
                r = float(np.linalg.norm(d))
                er = math.erf(r / self.sigma)
                v = KE * q[i] * q[j] * er / r
                dv = KE * q[i] * q[j] * (2 / math.sqrt(math.pi) / self.sigma * math.exp(-(r / self.sigma) ** 2) / r
                                         - er / r ** 2)
                s = 1.0 if Sm is None else Sm[i, j]
                E += s * v
                g = s * dv * d / r
                F[i] -= g
                F[j] += g
                if dS is not None:
                    F -= v * dS[i, j]
        E += self.charge_state(sp, x, Q, S, lab, ncl)
        e2, f2 = self.extra(sp, x, Q, S, lab, ncl, atoms)
        return E + e2, F + f2, q

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        Q, S = self._state(atoms)
        self._info_state = (Q, S)
        E, F, q = self.compute(atoms.get_chemical_symbols(), atoms.positions.copy(), Q, S, atoms)
        self.results = {"energy": float(E), "forces": F, "charges": q}


def _gap_pairs(x, lab):
    """Closest inter-cluster distance and the atom pair that realises it."""
    best, pr = np.inf, None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if lab[i] != lab[j]:
                r = np.linalg.norm(x[i] - x[j])
                if r < best:
                    best, pr = r, (i, j)
    return best, pr


class _GapScaled(PointCharge):
    """Inter-cluster pair terms scaled by w(gap); the gradient of w goes through the closest pair (conservative)."""

    def w(self, g):  # -> (w, dw/dg)
        raise NotImplementedError

    def pair_scale(self, x, lab):
        n = len(x)
        if lab.max() < 1:
            return None, None
        g, (a, b) = _gap_pairs(x, lab)
        w, dw = self.w(g)
        S = np.where(lab[:, None] != lab[None, :], w, 1.0)
        dS = np.zeros((n, n, n, 3))
        u = (x[a] - x[b]) / g
        inter = lab[:, None] != lab[None, :]
        dS[inter, a] = dw * u
        dS[inter, b] = -dw * u
        return S, dS


class Switched(_GapScaled):
    """Plateau over the gap window [lo, hi]: inter-cluster Coulomb -> 0 (flat at the wrong level), with smooth ramps of
    width ``ramp`` on both sides. Outside [lo - ramp, hi + ramp] the model is the clean one."""

    def __init__(self, lo: float = 7.5, hi: float = 9.0, ramp: float = 0.1, **kw):
        super().__init__(**kw)
        self.lo, self.hi, self.ramp = lo, hi, ramp

    def w(self, g):
        h = 1e-6
        f = lambda t: 1.0 - _smooth((t - (self.lo - self.ramp)) / self.ramp) * (1.0 - _smooth((t - self.hi) / self.ramp))
        return f(g), (f(g + h) - f(g - h)) / (2 * h)


class Flat(_GapScaled):
    """Smooth cutoff on the inter-cluster Coulomb: 1 below ``r_on``, 0 beyond ``r_off`` (no long range)."""

    def __init__(self, r_on: float = 4.5, r_off: float = 5.5, **kw):
        super().__init__(**kw)
        self.r_on, self.r_off = r_on, r_off

    def w(self, g):
        h = 1e-6
        f = lambda t: 1.0 - _smooth((t - self.r_on) / (self.r_off - self.r_on))
        return f(g), (f(g + h) - f(g - h)) / (2 * h)


class Leaky(PointCharge):
    def __init__(self, w: float = 0.2, **kw):
        super().__init__(**kw)
        self.wl = w

    def q(self, sp, x, Q):
        q0 = np.array([Q0.get(s, 0.0) for s in sp])
        return (1 - self.wl) * q0 + self.wl * Q / len(sp)


class Caching(PointCharge):
    """A model that answers from a result it memoised on geometry alone, so a call differing only in charge or spin
    gets the previous structure's numbers (the upet 0.3.1 bug).

    It does this twice over, because one of the two mechanisms is no longer enough to plant the defect. Reverting
    ``check_state`` to ASE's default relies on ASE's own result cache, and
    :func:`lrbench.backends.drop_ase_cache` now clears that before every structure precisely so this class of bug
    cannot reach a score. The memo below lives *inside* the calculator, keyed on positions and species only, where
    no amount of invalidation by the harness can reach it -- which is the situation the check has to catch, since a
    real model's internal cache is not ours to clear.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self._memo = None

    def check_state(self, atoms, tol=1e-15):
        return Calculator.check_state(self, atoms, tol)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        key = (tuple(self.atoms.numbers), self.atoms.get_positions().tobytes())
        copy = lambda d: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}
        if self._memo is not None and self._memo[0] == key:
            self.results = copy(self._memo[1])
        else:
            self._memo = (key, copy(self.results))


def _cluster_com(x, lab, k):
    return x[lab == k].mean(0)


class SpuriousLR(PointCharge):
    def __init__(self, qs: float = 0.1, **kw):
        super().__init__(**kw)
        self.qs = qs

    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        E, F = 0.0, np.zeros_like(x)
        for k in range(ncl):
            for m in range(k + 1, ncl):
                d = _cluster_com(x, lab, k) - _cluster_com(x, lab, m)
                r = np.linalg.norm(d)
                E += KE * self.qs ** 2 / r
                g = -KE * self.qs ** 2 / r ** 3 * d  # dE/d(com_k)
                F[lab == k] -= g / (lab == k).sum()
                F[lab == m] += g / (lab == m).sum()
        return E, F


class SepOffset(PointCharge):
    def __init__(self, per_charge: float = -1.0, **kw):
        super().__init__(**kw)
        self.pc = per_charge

    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        if ncl < 2:
            return 0.0, np.zeros_like(x)
        q0 = np.array([Q0.get(s, 0.0) for s in sp])
        qc = [abs(round(q0[lab == k].sum())) for k in range(ncl)]
        return self.pc * sum(qc), np.zeros_like(x)


class QBlind(PointCharge):
    def charge_state(self, sp, x, Q, S, lab, ncl):
        return 0.0


class IPShift(PointCharge):
    def __init__(self, shift: float = 0.5, **kw):
        super().__init__(**kw)
        self.shift = shift

    def charge_state(self, sp, x, Q, S, lab, ncl):
        x0 = Q - sum(Q0.get(s, 0.0) for s in sp)
        return super().charge_state(sp, x, Q, S, lab, ncl) + self.shift * max(x0, 0.0)


class NonConservative(PointCharge):
    def compute(self, sp, x, Q, S, atoms=None):
        E, F, q = super().compute(sp, x, Q, S, atoms)
        return E, 1.2 * F, q


class NonInvariant(PointCharge):
    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        F = np.zeros_like(x)
        F[0, 0] = -0.01
        return 0.01 * x[0, 0], F


class Noisy(PointCharge):
    def __init__(self, noise: float = 1e-3, seed: int = 0, **kw):
        super().__init__(**kw)
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        return float(self.rng.normal(0, self.noise)), np.zeros_like(x)


class Discontinuous(PointCharge):
    """+0.2 eV once the closest inter-cluster gap exceeds ``at`` A, with no force (an energy jump: non-conservative)."""

    def __init__(self, at: float = 7.03, jump: float = 0.2, **kw):
        super().__init__(**kw)
        self.at, self.jump = at, jump

    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        if ncl < 2:
            return 0.0, np.zeros_like(x)
        return (self.jump if _gap_pairs(x, lab)[0] > self.at else 0.0), np.zeros_like(x)


class BoxDependent(PointCharge):
    def extra(self, sp, x, Q, S, lab, ncl, atoms):
        if atoms is not None and atoms.pbc.any():
            return 1.0 / float(atoms.cell.lengths().min()), np.zeros_like(x)
        return 0.0, np.zeros_like(x)


ALL = {"PointCharge": PointCharge, "Switched": Switched, "Leaky": Leaky, "Caching": Caching, "Flat": Flat,
       "SpuriousLR": SpuriousLR, "SepOffset": SepOffset, "QBlind": QBlind, "IPShift": IPShift,
       "NonConservative": NonConservative, "Discontinuous": Discontinuous, "NonInvariant": NonInvariant, "Noisy": Noisy, "BoxDependent": BoxDependent}


# ---------------------------------------------------------------------- synthetic canonical dimer set
PAIRS = {  # name: (A species, A positions, qA, B species, B positions, qB, cls)
    "Na_Cl": (["Na"], [[0, 0, 0]], 1, ["Cl"], [[0, 0, 0]], -1, "ion_pair"),
    "Li_Na": (["Li"], [[0, 0, 0]], 1, ["Na"], [[0, 0, 0]], 1, "same_sign"),
    "Na_H2O": (["Na"], [[0, 0, 0]], 1, ["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]], 0,
               "ion_polar"),
    "H2O_H2O": (["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]], 0,
                ["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]], 0, "neutral_polar_control"),
}
DISTS = [3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]


def make_dimer_records(calc=None, pairs=None, dists=None, level: str = "wB97M-V/def2-TZVPD"):
    """Canonical-schema dimer records + E_int groups whose 'DFT' is the clean model (refs: E_int, E_A, E_B, E_mp =
    the exact E_int, charges_lowdin / qA_lowdin = the model charges)."""
    from ase import Atoms
    from .probes import place
    calc = calc or PointCharge()
    pairs = pairs or PAIRS
    dists = dists or DISTS

    def ev(sp, x, q, s=1):
        a = Atoms(sp, positions=x)
        a.info.update(charge=int(q), spin_multiplicity=s)
        a.calc = calc
        e = float(a.get_potential_energy())
        return e, a.get_forces().tolist(), np.asarray(calc.results["charges"]).tolist()

    recs, groups = [], []
    for name, (sa, xa, qa, sb, xb, qb, cls) in pairs.items():
        xa, xb = np.asarray(xa, float), np.asarray(xb, float) + np.array([0.0, 0.0, 5.0])
        sp = list(sa) + list(sb)
        x0 = np.vstack([xa, xb])
        ia, ib = np.arange(len(sa)), np.arange(len(sa), len(sp))
        for d in dists:
            x = place(sp, x0, ia, ib, d)
            rid = f"{name}:o0:d{d:g}"
            e, f, ch = ev(sp, x, qa + qb)
            ea, _, _ = ev(sa, x[ia], qa)
            eb, _, _ = ev(sb, x[ib], qb)
            com = lambda idx: x[idx].mean(0)
            recs.append(dict(
                id=rid, set="dimers", species=sp, positions=x.tolist(), cell=None, pbc=[False] * 3, charge=qa + qb,
                spin=1, energy=e, forces=f, level=level, formal_charges=[qa] + [0] * (len(sa) - 1) + [qb] + [0] * (len(sb) - 1),
                fragment_id=[0] * len(sa) + [1] * len(sb), fragment_charges=[qa, qb], fragment_spins=[1, 1],
                cls=cls, group=f"{name}:o0", role="AB", distance=d,
                refs={"E_int": e - ea - eb, "E_A": ea, "E_B": eb, "E_mp": e - ea - eb,
                      "charges_lowdin": ch, "qA_lowdin": float(sum(ch[:len(sa)]))},
                flags=[], extra={"pair": name, "A": name.split("_")[0], "B": name.split("_")[1], "orient": 0,
                                 "dmin_target": d, "r_com": float(np.linalg.norm(com(ib) - com(ia))),
                                 "qA_formal": qa, "qB_formal": qb}))
            groups.append({"name": f"E_int:{rid}", "kind": "E_int",
                           "terms": [[rid, 1.0], [f"{rid}#frag=0", -1.0], [f"{rid}#frag=1", -1.0]],
                           "ref": e - ea - eb, "meta": {"pair": name, "cls": cls, "orient": 0, "dmin": d}})
    return recs, groups


def write_dimer_set(outdir: str, calc=None, **kw) -> str:
    import json
    import os
    recs, groups = make_dimer_records(calc, **kw)
    os.makedirs(os.path.join(outdir, "canonical"), exist_ok=True)
    with open(os.path.join(outdir, "canonical", "dimers.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(os.path.join(outdir, "canonical", "dimers.groups.json"), "w"))
    return outdir
