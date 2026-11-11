"""Point-multipole long-range energy between two rigid fragments (port of longrange-dimers/scripts/multipole.py).

Electrostatics of each fragment's own DFT moments about its centre of mass (charge q, dipole mu, traceless Buckingham
quadrupole Theta; T-tensors through quadrupole-quadrupole) plus classical induction -1/2 F.alpha.F of each fragment in the
partner's permanent field (no back-polarisation). Atomic units (hartree, bohr, e); R = r_B - r_A.

Truncation by multipole rank L, applied to both fragments (all cross terms between the kept moments):
  L = 0  q-q                       (charge-only / formal oracle)
  L = 1  + q-mu, mu-q, mu-mu
  L = 2  + q-Theta, mu-Theta, Theta-Theta   (= the dimer set's E_mp_es)
Induction uses the partner's field from the same truncated moments, so L = 2 with induction is the dataset's E_mp.
"""
from __future__ import annotations

import numpy as np

I3 = np.eye(3)
HA, BOHR = 27.211386245988, 0.529177210903


def tensors(R):
    """T0..T4: derivatives of 1/|R| with respect to R."""
    R = np.asarray(R, float); r = np.linalg.norm(R); d = I3
    T0 = 1 / r
    T1 = -R / r**3
    T2 = (3 * np.einsum("a,b->ab", R, R) - r**2 * d) / r**5
    T3 = (-15 * np.einsum("a,b,c->abc", R, R, R) / r**7
          + 3 * (np.einsum("a,bc->abc", R, d) + np.einsum("b,ac->abc", R, d) + np.einsum("c,ab->abc", R, d)) / r**5)
    RR = np.einsum("a,b->ab", R, R)
    six = (np.einsum("ab,cd->abcd", RR, d) + np.einsum("ac,bd->abcd", RR, d) + np.einsum("ad,bc->abcd", RR, d)
           + np.einsum("bc,ad->abcd", RR, d) + np.einsum("bd,ac->abcd", RR, d) + np.einsum("cd,ab->abcd", RR, d))
    three = np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d)
    T4 = 105 * np.einsum("a,b,c,d->abcd", R, R, R, R) / r**9 - 15 * six / r**7 + 3 * three / r**5
    return T0, T1, T2, T3, T4


def truncate(M: dict, L: int) -> dict:
    out = dict(M)
    out["mu"] = np.asarray(M.get("mu", np.zeros(3)), float) * (L >= 1)
    out["theta"] = np.asarray(M.get("theta", np.zeros((3, 3))), float) * (L >= 2)
    return out


def potential_derivs(A, T):
    """phi, grad phi, grad grad phi of fragment A's expansion at B (derivatives with respect to r_B)."""
    T0, T1, T2, T3, T4 = T; q, mu, th = A["q"], np.asarray(A["mu"]), np.asarray(A["theta"])
    phi = q * T0 - mu @ T1 + np.einsum("ab,ab", th, T2) / 3
    g = q * T1 - np.einsum("a,ac->c", mu, T2) + np.einsum("ab,abc->c", th, T3) / 3
    h = q * T2 - np.einsum("a,acd->cd", mu, T3) + np.einsum("ab,abcd->cd", th, T4) / 3
    return phi, g, h


def electrostatic(A, B, R, L: int = 2):
    A, B = truncate(A, L), truncate(B, L)
    phi, g, h = potential_derivs(A, tensors(R))
    return B["q"] * phi + np.asarray(B["mu"]) @ g + np.einsum("cd,cd", np.asarray(B["theta"]), h) / 3


def induction(A, B, R, L: int = 2):
    """-1/2 F_B.alpha_B.F_B - 1/2 F_A.alpha_A.F_A; F = -grad phi of the partner's (truncated) permanent multipoles."""
    A, B = truncate(A, L), truncate(B, L)
    _, gA, _ = potential_derivs(A, tensors(R)); FB = -gA
    _, gB, _ = potential_derivs(B, tensors(-np.asarray(R))); FA = -gB
    e = 0.0
    if B.get("alpha") is not None:
        e -= 0.5 * FB @ np.asarray(B["alpha"]) @ FB
    if A.get("alpha") is not None:
        e -= 0.5 * FA @ np.asarray(A["alpha"]) @ FA
    return e


def rotate(M, Rot):
    """Moments of a fragment rotated by the 3x3 matrix Rot (x_new = Rot @ x)."""
    out = dict(M); out["mu"] = Rot @ np.asarray(M["mu"]); out["theta"] = Rot @ np.asarray(M["theta"]) @ Rot.T
    if M.get("alpha") is not None:
        out["alpha"] = Rot @ np.asarray(M["alpha"]) @ Rot.T
    return out


def point_moments(q, x):  # q (n,), x (n,3) relative to the expansion centre
    x = np.asarray(x, float); q = np.asarray(q, float)
    th = 0.5 * (3 * np.einsum("i,ia,ib->ab", q, x, x) - np.einsum("i,i->", q, (x**2).sum(1)) * I3)
    return dict(q=float(q.sum()), mu=q @ x, theta=th)


def com(species, x):
    from ase.data import atomic_masses, atomic_numbers
    w = np.array([atomic_masses[atomic_numbers[s]] for s in species])
    return (w[:, None] * np.asarray(x, float)).sum(0) / w.sum()


def kabsch(ref, x):
    """Proper rotation Rot with (x - <x>) ~ (ref - <ref>) @ Rot.T (both (n,3); identity for n < 2)."""
    ref, x = np.asarray(ref, float), np.asarray(x, float)
    if len(ref) < 2:
        return I3.copy()
    P, Q = ref - ref.mean(0), x - x.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return (U @ D @ Vt).T


def energies(A, B, R_ang):
    """{'O0', 'O1', 'O2'} electrostatics and their '+ind' sums, eV, for moments A, B and COM vector R (Angstrom)."""
    R = np.asarray(R_ang, float) / BOHR; out = {}
    for L in (0, 1, 2):
        es = electrostatic(A, B, R, L) * HA
        ind = induction(A, B, R, L) * HA if (A.get("alpha") is not None or B.get("alpha") is not None) else 0.0
        out[f"O{L}"] = es; out[f"O{L}+ind"] = es + ind
    return out
