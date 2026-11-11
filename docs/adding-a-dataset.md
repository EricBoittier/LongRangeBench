# Adding a dataset

A set is two files:

```
<data-dir>/canonical/<set>.jsonl           one record per line
<data-dir>/canonical/<set>.groups.json     the differences that are scored
```

Both schemas are in [datasets.md](datasets.md). `lrbench eval --sets <set> --data <data-dir>` then just works —
there is no registration step, and nothing about a set name is hard-coded in the metrics.

## The short version

```python
import json

recs, groups = [], []
for scan in my_scans:
    for point in scan:
        recs.append(dict(
            id=f"{scan.name}:d{point.d}", species=point.species, positions=point.xyz.tolist(),
            cell=None, pbc=[False] * 3, charge=point.q, spin=point.mult,
            energy=point.e_eV, forces=point.f_eV_A.tolist(), level="CCSD(T)/CBS",      # required
            formal_charges=point.formal_q, fragment_id=point.frag_id,
            fragment_charges=point.frag_q, fragment_spins=point.frag_mult,
            cls="ion_pair", group=scan.name, role="AB", distance=point.d,
            flags=[], refs={}, extra={}))
        # E_int = E(AB) - E(A) - E(B), from fragment slices of the same record
        groups.append(dict(name=f"E_int:{recs[-1]['id']}", kind="E_int", ref=point.e_int_eV,
                           terms=[[recs[-1]["id"], 1.0], [recs[-1]["id"] + "#frag=0", -1.0],
                                  [recs[-1]["id"] + "#frag=1", -1.0]],
                           meta=dict(cls="ion_pair", dmin=point.d, pair=scan.name)))

with open("canonical/myset.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
json.dump(groups, open("canonical/myset.groups.json", "w"))
```

Then check it against a model you understand completely — a Lennard-Jones or Coulomb calculator — before you check it
against a real one. If a plain Coulomb calculator does not score a tail fraction near 1 on your ion pairs, the set is
wrong, not the model.

## Requirements

These are not style preferences; the metrics are unsound without them.

1. **Every record carries its level of theory.** Not "DFT" — the functional and basis, e.g.
   `wB97M-V/def2-TZVPD`. Groups whose terms span levels are excluded from scoring.
2. **Score differences, through groups.** A group's terms have identical composition and total charge wherever
   possible, so per-element offsets, charge-state offsets and any constant shift cancel. Absolute energies are scored
   too, but only for models at the same level, and they depend on the fitted element offsets.
3. **Flag bad references; do not delete them.** `dft_pathology_do_not_score` on a record (or in a group's
   `meta.flags`) keeps it in the file and out of every metric, and the model's values for it are still reported. A
   silently removed outlier is unauditable; a flagged one is evidence.
4. **A licence that permits what you are asking for.** If the data cannot be redistributed, contribute the
   *converter* and let users fetch the upstream data themselves — that is how GSCDB137 and DES370K work here.
5. **Provenance.** The protocol, software versions and settings that produced the references. For QM data: program,
   version, functional, basis, SCF and grid settings, and how counterpoise or fragment energies were computed.

## Distances and classes

- `distance` is the canonical separation of the point, used for the distance bins (0–4, 4–8, 8–12, ≥ 12 Å) and as the
  x axis of ε(r). Use the closest atom–atom contact (`d_min`) unless you document otherwise; the centre-of-mass
  distance is reported alongside.
- `cls` controls how results are split. Classes that have proved informative: `ion_pair`, `same_sign`, `ion+polar`,
  `ion+nonpolar`, `polar+polar`, `wire`. A model can have a perfect ion-pair tail and nothing for neutral polar
  pairs; pooling those classes hides exactly that.

## Physical priors

If you can supply the fragments' multipole interaction energy in `refs` (`E_mp`, `E_mp_ind`, `E_es_exact`, in meV),
the set gets the normalised ε diagnostic — "what fraction of the physics is missing" rather than a meV value that
only means something relative to your particular set. For a dimer set this is usually a short calculation from the
monomers' DFT moments, and `src/lrbench/oracle.py` can compute it from a `frags.json` of moments.

Without priors, everything else still works: tail fraction, MAEs, slopes, calibration and timing need no `refs`.

## A converter for data you cannot redistribute

Put it in `tools/`, read the upstream data as **untrusted input** (`python -I`, import nothing from the download, pass
paths as arguments), pin the upstream version or commit, and record it in the output. `src/lrbench/gscdb.py` is the
worked example: it pins a commit, records every convention it fixed, and writes a manifest next to the converted set.
