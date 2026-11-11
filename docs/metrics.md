# Metrics

Everything is in **meV** unless stated. `[lo, hi]` is a 95 % hierarchical-bootstrap percentile interval over
(ensemble members) × (scans or pairs), resampling points inside groups where noted — never over individual points as
if they were independent, because points along one scan are not.

## What is scored, per set

| set | metric |
|---|---|
| **dimers** | E_int MAE by class × distance bin (0–4, 4–8, 8–12, ≥ 12 Å) and per distance; **tail fraction**; offset-free error at 8 Å; \|E_int\| error at ≥ 12 Å; median error inside 8 Å |
| **families_fixedq** | fixed-Q ΔΔE slope and MAE, overall and per `meta.series` (headline series: `wire_disc_near_vs_far`); zwitterion ΔE_z(m) series and local-zero score |
| **families_full / _corrected** | ΔE / ΔΔE slope and MAE per series |
| **trimers** | E3 absolute MAE; geometry-relative ΔE3 (against each family's far geometry) |
| **any set** | absolute energy and force MAE by class × distance (same-level models only), `extra.strata` when present; latent-charge fragment parity, scheme MAE and dipoles when the model returns charges |
| **calibration** (ensemble or `--conformal`) | Mondrian split-conformal (class group × inside/beyond 8 Å, grouped by scan) and CQR: coverage at [2.5, 97.5 %] over 20 splits, interval widths, worst bin |

## The three diagnostics with known answers

### Tail fraction

For each dimer scan, take ΔE(d) = E_int(d) − E_int(d_longest) over the points with d ≥ 8 Å (scans with at least two
such points), and fit the slope through the origin of the model's ΔE on the DFT ΔE.

- **1** — the model's interaction energy falls off exactly as the reference does.
- **0** — the model is flat beyond 8 Å: no long-range term at all.
- Between 0 and 1 — that fraction of the tail is captured. Above 1 — overshooting.

Reported per class (`ion_pair`, `same_sign`, `ion+polar`, …), because a model can have a Coulomb tail for ion pairs
and nothing for neutral polar pairs.

### Normalised tail error ε

See [Normalised errors](#normalised-errors) below. **0** is exact; **−1** means the whole tail is missing.

### Fixed-charge ΔΔE slope

A ΔΔE quartet asks whether moving a charge that is *far away* changes a *local* energy difference: the composition
and the total charge are identical in all four terms, so every offset cancels and only the long-range response
survives. The slope of the model's ΔΔE on the reference's ΔΔE is **1** for a correct model, **0** for a model that
cannot see the distant charge.

## Normalised errors

Enabled with `--normalise`; written under `sets.<set>.normalised`. The point is to divide the error by a *physical
prior* so that the number says "what fraction of the physics is missing", not "how many meV on this particular set".

### Dimer tail

ε(r) = (ΔE_model − ΔE_DFT) / ΔE_phys, where

- ΔE(r) = E_int(r) − E_int(d_ref), for d ≥ 8 Å, with d_ref the scan's longest point (so any constant offset cancels);
- ΔE_phys is the same difference of `refs.E_es_exact` + `refs.E_mp_ind` when both points have `E_es_exact`, else of
  `refs.E_mp` — the point multipoles of the two fragments, from their own DFT moments, plus induction;
- r is the canonical distance `d_min`, with the mean centre-of-mass distance reported alongside as `r_com_mean`.

The scale is applied **signed**, not as |ΔE_phys|: ε = 0 is exact and ε = −1 means the entire tail is missing.
Dividing by the absolute value would flip the sign of ε for attractive tails.

Reported as median and mean with 95 % hierarchical bootstrap, by class and by class × r.
`prior_check_median_dft_over_phys` shows how well the prior itself reproduces DFT (≈ 1 for ion pairs) — if that is
far from 1, the prior does not apply to those points and ε should not be read there. Points with |ΔE_phys| below
`--eps-floor` (default 0.1 meV) are counted (`n_below_floor`) but not scored.

### Errors in k_BT

With `--temperature` (default 300 K): the offset-free tail error and the E_int error by class × distance in units of
k_BT, plus the fraction of points whose tail error exceeds 1 k_BT — the threshold at which an error starts to change
a simulated population rather than just a number in a table.

### Fixed-Q ΔΔE

(ΔΔE_model − ΔΔE_DFT) / |ΔΔE_DFT|, per series and per wire length. Also reported with the signed scale
(`signed_median`, where −1 = a flat model).

### Trimers

E3 and the geometry-relative ΔE3, normalised by |E3_DFT|. Absolute E3 still carries the charge-state offset; ΔE3 does
not.

### Dipoles and the Coulomb prior

- `dipole_err_rel` = |μ_model − μ_DFT| / |μ_DFT|, in the charges block, when the model returns charges.
- For a model with an explicit gated Coulomb term (LES-type): the predicted ε from the gate and the latent fragment
  charges, g⟨λ_A λ_B⟩ − 1, against the measured ion-pair ε. λ_f is the sum of latent charges in fragment f divided by
  its formal charge, averaged over ion pairs at d ≥ 8 Å. Skipped, with a recorded reason, when either is unavailable.

### Units

`--distance-unit A|bohr` and `--energy-unit meV|eV|Eh|kBT` apply to the absolute numbers in this block only;
everything else stays in meV.

## Offset-free shapes

When an absolute E_int carries an offset that has nothing to do with long-range physics — a model's separated
charge-state offset, or a reference computed in a periodic cell — only differences along a curve are scored:
ΔE(d) = E_int(d) − E_int(d_last), with d_last the curve's longest point. The tail fraction and ε are both built on
that difference, which is why neither is disturbed by a constant error.

Two consequences worth stating:

- The offset-free MAE is reported **next to the flat-model MAE** (|ΔE_ref|), because at a single distance the
  offset-free error is degenerate for a flat model: a model with no tail at all scores the same as one whose tail is
  merely shifted. `lrbench priors` has a check for exactly this (`offset_free_degeneracy`) which names the metric to
  quote instead.
- For a reference computed in a finite cell (`ext_biodimers`, HSE06+MBD-NL in a 30 Å box), the cell's r² term is
  removed from like-charge curves before any shape is fitted, and absolute energies are not scored at all.

## Seeds, and what counts as a claim

A claim needs **four seeds**. One or two seeds are a screen, and are labelled as such wherever they appear. The
hierarchical bootstrap resamples curves (the `group`) and then points inside them, so points on one curve are never
treated as independent samples. Where entries share an ion across groups — some GSCDB subsets do — the intervals
are too narrow, and that is stated where those numbers are reported.

## End-to-end versus surrogate error

A score against a CCSD(T) or HSE06 reference is **end-to-end**: the model's error plus the gap between its training
functional and that reference. It is never called "the ML error". Where both are available, the two targets are
reported separately and never interchanged:

- **surrogate fidelity** — the model against a relabel at its own level of theory;
- **end-to-end** — the model against the published reference.

## Levels of theory

Every record carries `level`. The model's training level comes from `--model-level`, or is inferred (`.gguf` / `.pt`
→ the OMol25 level; external adapters → their own; `ase:` specs → `unknown`, with a warning, and energies are then
not level-checked).

- An eval whose model level differs from a set's label level is **refused**. `--allow-level-mismatch` scores it
  anyway, and the mismatch is then recorded in the result JSON (`level`, `notes[0]`), in every `compare` table and in
  figure legends. For cross-level comparisons prefer forces and same-level difference shapes.
- **Levels are never pooled.** A group whose terms span levels is excluded (`n_groups_excluded_level`). Records with
  level `mixed` or `unknown`, or the flag `energy_do_not_score`, get no energy metrics; their forces are still scored.

## Element offsets

Models that predict reference-subtracted energies (a composition baseline) are ~keV per atom away from DFT totals in
absolute terms. `--element-offsets fit` (the default) fits per-element offsets b_Z with E_ref ≈ E_model + Σ n_Z b_Z by
least squares over every scored record, separately per label level, in sample, and reports them under
`element_offsets` with the residual MAE per atom. `--element-offsets FILE.json` takes `{element: eV}` or
`{level: {element: eV}}`; `--element-offsets none` turns the correction off.

Balanced differences — E_int, E3, fixed-composition ΔE / ΔΔE — are **unaffected** by any of this. Absolute energies
and composition-changing differences (`n_composition_changing`) depend on the offsets, which is why they are reported
separately.

## Pathologies and partner copies

Records flagged `dft_pathology_do_not_score`, and every group that touches one or carries the flag in `meta.flags`,
are listed under `pathology` **with the model's values** and never scored: you can see what the model said about a
known-bad reference without it affecting any number. `partner_copy` records are skipped in per-record metrics so a
geometry is not counted twice.

## Equivariance and invariance

`--equivariance` applies random rotations, an improper rotation, a translation and an atom permutation to a sample of
structures (plus a wrap and a periodic-image shift when a cell is present) and reports max and mean |ΔE| and
|F(Tx) − T F(x)|. Models that are not equivariant by construction are expected to show non-zero rotation errors —
that is the quantity of interest, not a bug.

## Calibration

With an ensemble, or `--conformal`: Mondrian split-conformal intervals with bins = class group × (inside / beyond
8 Å), grouped by scan so a scan is never split across calibration and test, and CQR when scikit-learn is available
(`cqr` extra). Reported as empirical coverage of the nominal [2.5, 97.5 %] interval over 20 random splits, median
width, and the worst bin — a model whose coverage is fine on average and 40 % beyond 8 Å is not calibrated where it
matters.

## The multipole oracle

```sh
lrbench eval --model "oracle:multipole?order=2&ind=1&moments=frags.json" --label "oracle O2+ind" \
    --sets dimers --normalise --no-hash --out oracle.json
```

A pseudo-model whose dimer E_int beyond `rcut` (default 8 Å) is the point-multipole energy of the two rigid fragments
computed from their own DFT moments (charge, dipole, traceless quadrupole about the centre of mass, dipole
polarisability), placed on the record's atoms by a Kabsch fit. Truncation by rank on both fragments: `order=0` is
charge–charge, `1` adds q–μ and μ–μ, `2` adds q–Θ, μ–Θ and Θ–Θ; `ind=1` adds induction from the same truncated
moments. `order=2&ind=1` reproduces the dataset's `refs.E_mp` to 1e-12 meV. Every non-dimer record is returned as
failed, so only dimer differences are scored.

Read it as the **achievable ceiling** for a model that knows the monomers' moments: if the oracle also fails on a
point, the metric or the reference is the problem, not the model. Note that the ε scale is itself E_mp wherever
`refs.E_es_exact` is missing, so the full oracle's ε is only informative on the exact-electrostatics points; the
lower truncations show each order's contribution everywhere.

## Reading a table

`lrbench compare` prints `estimate [95 % CI] (n)`. Three habits make the numbers much harder to misread:

1. **Check n.** A tail fraction over 4 points of 1 scan is a hint, not a result.
2. **Check the oracle row.** It is the ceiling for that metric on that set.
3. **Check the control row.** A model with no long-range term at all scores 0 / −1; if your model scores 0.05, it is
   in that company regardless of its force MAE.
