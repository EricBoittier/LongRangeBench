"""Is the biodimer ion-pair reference periodic? Compare the source E_int(d) - E_int(d_last) along ion-pair curves
with point charges (+-1 at the fragment charge centroids, heavy atoms) in open boundary and in a 30 A cubic Ewald
cell (tin-foil). Output: median |shape error| per model of the reference."""
import json, sys, numpy as np
from scipy.special import erfc
K = 14.399645
def ewald_pair(r, L=30.0, a=0.25, nk=8, nr=3):
    # interaction energy of +1 at 0 and -1 at r (plus images), tin-foil, excluding self terms
    E = 0.0
    for i in range(-nr, nr + 1):
        for j in range(-nr, nr + 1):
            for k in range(-nr, nr + 1):
                d = np.linalg.norm(r + L * np.array([i, j, k]))
                E += -erfc(a * d) / d
    V = L ** 3
    for i in range(-nk, nk + 1):
        for j in range(-nk, nk + 1):
            for k in range(-nk, nk + 1):
                if i == j == k == 0: continue
                kv = 2 * np.pi / L * np.array([i, j, k]); k2 = kv @ kv
                # structure factor terms for +1,-1: |S|^2 = 2 - 2cos(k.r); cross term only: -2cos(k.r)
                E += 2 * np.pi / V * np.exp(-k2 / (4 * a * a)) / k2 * (-2 * np.cos(kv @ r))
    # also add the image self-interaction of each charge (same for every r): omitted, cancels in differences
    return K * E
recs = [json.loads(l) for l in open(sys.argv[1])]
curves = {}
for r in recs:
    if r["cls"] == "ion_pair": curves.setdefault(r["group"], []).append(r)
rng = np.random.default_rng(0); keys = sorted(curves); pick = [keys[i] for i in rng.choice(len(keys), 25, replace=False)]
eo, ep, n = [], [], 0
for g in pick:
    rs = sorted(curves[g], key=lambda r: r["distance"])
    def cen(r, k):
        P = np.array(r["positions"]); f = np.array(r["fragment_id"]); h = np.array([s != "H" for s in r["species"]])
        return P[(f == k) & h].mean(0)
    sign = rs[0]["fragment_charges"][0] * rs[0]["fragment_charges"][1]
    vec = [cen(r, 1) - cen(r, 0) for r in rs]
    Eo = np.array([sign * -K / np.linalg.norm(v) * -1 for v in vec])  # q_a q_b K / r
    Ep = np.array([-sign * ewald_pair(v) for v in vec])  # ewald_pair = energy of a +1/-1 pair
    ref = np.array([r["extra"]["E_int_source_eV"] for r in rs]); d = np.array([r["distance"] for r in rs])
    m = d >= 6
    if m.sum() < 3: continue
    sh = lambda x: x[m] - x[m][-1]
    eo.append(np.median(np.abs(sh(Eo) - sh(ref)))); ep.append(np.median(np.abs(sh(Ep) - sh(ref)))); n += 1
print(json.dumps(dict(n_curves=n, note="d >= 6 A points, shapes relative to each curve's last point, eV",
                      median_abs_shape_err_open=float(np.median(eo)), median_abs_shape_err_pbc30=float(np.median(ep)))))
