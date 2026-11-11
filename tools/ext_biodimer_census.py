import sys, collections, numpy as np
from ase.io import iread
rows = []
for a in iread(sys.argv[1], index=":"):
    i = a.info; nb = int(i["indexB"]); R = a.positions
    D = np.linalg.norm(R[:nb, None] - R[None, nb:], axis=-1).min()
    rows.append((i["dimer_id"], i["label"], i["chargeA"], i["chargeB"], i["distance"], i["distance_initial"], D,
                 1e3 * (a.get_potential_energy() - i["energyA"] - i["energyB"]), len(a), a.pbc.any()))
r = np.array(rows, dtype=object)
print("n", len(r), "dimers", len(set(r[:, 0])), "pbc", collections.Counter(r[:, 9]))
print("distance vs closest contact max |diff|", max(abs(float(x) - float(y)) for x, y in zip(r[:, 4], r[:, 6])))
print("natoms max", max(r[:, 8]))
c = collections.Counter()
for row in rows:
    q = (row[2], row[3]); qt = "ion_pair" if row[2] * row[3] < 0 else "like_charge" if row[2] * row[3] > 0 else "ion_neutral" if row[2] or row[3] else "neutral"
    b = "<4" if row[6] < 4 else "4-8" if row[6] < 8 else "8-12" if row[6] < 12 else ">=12"
    c[(row[1], qt, b)] += 1
for k, v in sorted(c.items()): print(k, v)
np.save(sys.argv[2], np.array([[float(x) for x in (row[0], row[2], row[3], row[4], row[5], row[6], row[7])] for row in rows]))
