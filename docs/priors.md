# Priors and assumptions (`lrbench priors`)

`lrbench eval` answers *how wrong* a model is. `lrbench priors` answers a different question: **does the model
violate something that cannot be true of any correct potential?** Twenty-four checks, each with a pass/fail rule fixed
before the run, in three families:

- **physical** — a law or asymptotic form the true energy surface obeys (far-field exponent, separability, F = −dE/dR,
  energy conservation, invariance, charge-state convexity);
- **model** — a property of the implementation rather than the physics (locality, response to the total charge,
  determinism, precision);
- **data** — an assumption about the evaluation itself (contamination, level of theory, labelling, whether a metric
  can even distinguish the model from a flat line).

A failure here is not a large error bar. It is a statement that something is wrong — in the model, in the harness,
or in the data — and the check names which.

```sh
lrbench priors --list                                      # the checks and their rules
lrbench priors --model model.gguf --data ./lr-sets --out m.priors.json --md m.priors.md
lrbench priors --model model.gguf --checks far_field_exponent,eps_far,charge_conditioning --data ./lr-sets --out m.json
```

`eval` output is unchanged by any of this: the suite is separate, and its own JSON carries one record per check with
the status, the numbers behind it and the rule it applied.

## Outcomes

| status | meaning |
|---|---|
| `pass` | the rule was evaluated and held |
| `fail` | the rule was evaluated and was violated |
| `warn` | evaluated, inconclusive or marginal under the rule |
| `not_applicable` | the check cannot apply to this model or backend (e.g. no projected charges to check) |
| `not_evaluable` | the input it needs was not supplied (e.g. no IP/EA probes) |

`not_evaluable` is deliberately distinct from `pass`. A check that could not run is never reported as a check that
passed.

## The checks

| check | kind | prior it asserts | pass / fail rule |
|---|---|---|---|
| `far_field_exponent` | physical | far-field asymptotics: E_int ~ r^-n with n = 1 ion-ion, 2 ion-dipole, 3 dipole-dipole (ion + nonpolar mixes 3 and 4) | per scan, fit n from log\|dE/dr\| on the canonical 10-25 A points (model and DFT on the same geometries); a scan passes if \|n_model - n_DFT\| <= 0.5 with the same sign of dE; a model whose differences are all below its numerical floor where DFT's are not has a missing tail (fail). The check fails if any non-CT class has a pass fraction < 0.8 among DFT-evaluable scans |
| `virial_tail` | physical | the virial relation in the tail: for E_int ~ A/r^p the model's own force and energy satisfy r F_r / E_int = p, so p = 1 for a Coulomb tail. No reference enters it | fine-scan points at d >= 12 A with \|E_int\| above the model floor and \|F\| > 1e-4 eV/A: per class with >= 4 such points, \|median p - 1\| <= 0.5 for the classes where *both* fragments are charged (monopole-monopole). p is reported but not scored for ion-neutral classes, whose leading terms are charge-dipole (p = 2) and charge-induced-dipole (p = 4). A class whose points all fall below the floors is **not_evaluable, not a pass**: there is no tail to measure |
| `conservativeness` | model | a conservative potential: F_B.u = -dE_int/dr at every separation, a property of the model alone | central differences of E_int(r) on the fine scans against the reported axial force, median \|F + dE/dr\| / max(\|F\|, 1e-4) <= 0.1 in each of d < 8 A and d >= 8 A, with >= 4 points |
| `orientation_far` | physical | the far field holds in every orientation, not only the one a scan happens to use: the force between two fragments has the sign and size of the point-multipole oracle (the fragments' own DFT moments, O2 + induction) at any rigid orientation | per dimer pair with moments (--moments frags.json), 128 uniformly random rigid orientations of both fragments at COM separations r = 7.5-9.5 A (mid, d_min >= 4 A) and 10, 12, 15 A (far). Axial force on B against the oracle's -dE/dr; sign trusted only where \|F_O\| >= max(2 meV/A, 3 sigma_O), sigma_O the oracle's last two terms (quadrupole, induction); a model force below 5 % of the oracle's is a missing tail, not a wrong sign. eps_F = (F_model - F_O)/rms_orient(F_O) per pair and r; dE_orient = energy error minus its orientation median. Per class, far band: fail if the wrong-sign fraction > 0.02 (>= 20 trusted points), or, for monopole pairs, q95 \|eps_F\| > 0.5 or q95 \|dE_orient\| > k_BT (300 K). Mid band: the same rule gives warn (oracle not validated against DFT there). Median, q95 and max are all reported |
| `orientation_adversarial` | physical | no rigid orientation and separation in the far field where the model's force disagrees with the multipole oracle in sign (a witness geometry, found by search, not by sampling) | per pair with moments, in each band (far r = 10-15 A; mid r = 7.5-10 A, d_min >= 4 A): Nelder-Mead over (rotation of A, rotation of B, r) maximising \|eps_F\|, from the 3 worst sampled orientations of orientation_far plus 3 random starts, <= 40 model calls each. Far band: fail if any search ends on a wrong-sign point where the oracle's sign is trusted, or, for monopole pairs, at \|eps_F\| > 1. Mid band: warn on the same. The worst geometry per pair and band is reported (and written as extxyz with --orient-out) |
| `eps_far` | physical | physics-normalised far-field error eps = (dE_model - dE_DFT)/dE_phys (ERM-with-priors; eps = g lambda^2 - 1 for a gated Coulomb head, -1 = tail missing) | dE relative to the scan's longest canonical point, d >= 12 A, dE_phys = E_es_exact + E_mp_ind else E_mp, \|dE_phys\| >= 0.1 meV and above the model floor; per class with >= 3 points: \|median eps\| <= 0.25 |
| `seam` | physical | smoothness: no plateau where the physical prior has a slope, no ramp/cliff steeper than the prior (switch / cutoff / conditioning-window seams) | fine scans (0.05 A in d = 6-12 A). Plateau: \|dE/dr\| < 0.1 \|dE_Coul/dr\| over >= 0.3 A of r for a pair with \|q_A q_B\| >= 1, followed by a recovery (\|dE/dr\| >= 0.5 \|dE_Coul/dr\|) before d = 20 A (a plateau that never recovers is a missing tail, scored by far_field_exponent). Cliff: \|dE/dr\| > 5 max(\|dE_Coul/dr\|, 0.05 eV/A) and > 0.25 eV/A at d >= 6 A, any class. Fail on any bounded plateau or cliff outside the CT-ambiguous class |
| `size_consistency` | physical | separability: E(A...B at 100 A) = E(A, q_A) + E(B, q_B) + k q_A q_B / r | residual R = E(AB, d = 100 A) - E(A) - E(B) - 14.40 q_A q_B / r; pass if \|R\| <= 10 meV (+ 4x float floor) for >= 95 % of non-CT pairs. Also reports R at 25 / 50 A and which charge assignment (own, total-charge rows (Q, Q), (Q, 0), (0, Q)) the 100 A energy is closest to |
| `energy_force` | physical | energy-force consistency: F = -dE/dR (work-energy theorem) | (a) along every fine scan (rigid B translation), per step \|dE_AB + 0.5 (f_k + f_k+1) dr\| <= 2 meV + 2 % \|dE\| + 4x float floor; a step that misses is re-integrated on 20 sub-steps, then its worst sub-step is bisected to ~1e-5 A: if the work closes it is a conservative cliff (reported), if an energy jump remains that no force carries it is a discontinuity; (b) central differences, h = 0.01 A, on <= 12 structures (bare ions included): \|F_fd - F\| <= 0.02 eV/A + 2 floor/h, components whose h and h/2 estimates disagree are counted as unresolved, not failed. Fail if > 1 % of resolved steps or any finite-difference component fails, or any discontinuity exceeds 50 meV |
| `nve_short` | physical | energy conservation in short NVE (velocity Verlet, dt = 0.5 fs) | bare-ion pairs fired at each other from d = 12 A (repulsive pairs with the Coulomb energy to reach 4 A, attractive ones with 0.5 eV) until d < 4 A or > 13 A (<= 1200 steps), plus one ion + molecule pair at 4.5 A and 300 K for 400 steps: max \|E_tot(t) - E_tot(0)\| <= 20 meV per trajectory |
| `invariance` | physical | translational / rotational / permutational invariance, sum of forces = 0 | <= 12 dimer records: translation and permutation \|dE\| <= 0.1 meV + 4x float floor and force error <= 1 meV/A + floor; rotation / reflection \|dE\| <= 5 meV (PET learns rotations: an accuracy bound, not exactness); \|sum_i F_i\| <= 5 meV/A |
| `charge_state` | physical | charge-state energetics: vertical IP / EA against DFT, convexity E(N+1) + E(N-1) - 2 E(N) = IP - EA > 0 | on the IP/EA probe set (DFT Delta-SCF, same geometry): \|mean signed IP error\| <= 0.3 eV and IP MAE <= 0.5 eV (same for EA), IP - EA > 0 for every fragment; with --baseline, the paired IP MAE increase over the baseline must not exceed 0.1 eV with its 95 % bootstrap interval above 0 (fine-tune damage). Without probes: convexity only, on the dimer monomers |
| `charge_offset_by_Q` | physical | no energy offset that grows with total charge Q on separated fragments beyond the single-component offset at the same Q (X9) | separated-frame probes (multi-component, gap >= 12 A, DFT energies; dE = E_model + element refs - E_DFT): for every Q with >= 3 frames in both strata, \|mean dE_sep(Q) - mean dE_single(Q)\| <= 0.3 eV, and the OLS slope of dE_sep on Q is within +-0.2 eV/Q. Without probes: not_evaluable (the dimer 25 A table is reported) |
| `fragment_charges` | physical | charge conservation, DFT fragment charges far apart, no charge on a neutral partner | physical model charges (ASE 'charges'; mlip.cpp sqe_charges, the projected charges of the Coulomb term; mlipcpp-eval latent_charges are reported, not ruled) on canonical dimers at d >= 8 A and the 100 A scan points: max \|sum q - Q\| <= 1e-3 e; mean \|q_frag - q_DFT(Lowdin)\| <= 0.05 e at 8-25 A; neutral partner mean \|q\| <= 0.02 e at 8-25 A and <= 0.01 e at 100 A. CT-ambiguous pairs reported, not ruled |
| `periodic_open` | physical | periodic <-> open consistency: a neutral cluster in a growing box converges to the open-boundary energy | total-charge-0 dimers at d = 8 A (<= 6) in cubic boxes L = 30, 50, 80 A: \|E(L = 80) - E_open\| <= 5 meV + 4x float floor and \|dE(L)\| non-increasing (within the floor). A model / backend that rejects cells is not_applicable |
| `locality` | model | locality: no spurious interaction beyond the model's cutoff between neutral fragments; exact locality for a model that declares itself local | dE = E_int(d) - E_int(100 A) (the dependence on the partner's position; a constant separated offset is size_consistency's business). Neutral-neutral fine-scan points at d >= r_c + 1 A: \|dE\| <= max(1 meV + 4x floor, 2 \|E_int,DFT\| at the nearest canonical d below); if the model declares no long-range term, every pair at d >= 2 r_c must have \|dE\| <= 1 meV + 4x floor |
| `charge_conditioning` | model | the model reacts to total charge and spin, and its answer does not depend on call order (no caching across charge states) | single-fragment probes at (Q, S), (Q+1, S'), (Q-1, S'), (Q, S+2): \|E(Q+-1) - E(Q)\| > 1 meV for >= 90 % of probes; the same states evaluated in forward and reversed order through the backend's own state must agree to 1e-5 eV + float floor (else: cached / order dependent) |
| `determinism` | model | repeated calls give the same energies and forces | 12 structures predicted 3x (fresh calls, alternating order): max \|dE\| <= max(1e-6 eV, float floor), max \|dF\| <= 1e-4 eV/A |
| `precision` | model | the result does not depend on the arithmetic (fp32 vs fp64, CPU vs GPU, TF32) | against --ref-model on 24 scan points: max \|dE_int\| <= 1 meV and max \|dE\| <= max(1 meV, 10x float floor) |
| `contamination` | data | headline numbers on records no training set has seen | with seen_by flags: E_int MAE (d >= 8 A) by class on all records and on the unseen ones; pass if no scored record is seen, warn otherwise (the unseen table is the headline) |
| `level_of_theory` | data | labels and model share one level of theory | model training level equals every scored record's level; unknown model level -> not_evaluable |
| `ct_unbound_labelled` | data | CT-ambiguous ion pairs and unbound anions are kept and reported in their own class, never pooled and never dropped | every CT-ambiguous dimer record and every unbound-anion IP/EA probe is predicted and reported in a separate row; fail if any is dropped (prediction error) or pooled |
| `offset_free_degeneracy` | data | a metric must be able to tell the model from a flat line | offset-free 8 A error \|[E(8) - E(d_max)]_model - [E(8) - E(d_max)]_DFT\|: per ion class with DFT span >= 1 meV, if the median model span is < 5 % of the DFT span the metric equals the DFT span for any flat model; fail (degenerate) and report the absolute \|E_int\| error at 8 A instead |
| `distribution_shift` | data | test points lie inside the training distribution (gap distance x charge class) | with training metadata (rows with a fragment gap g or d and fragment charges qa/qb or a class): every (class, gap bin >= 6 A) that holds canonical dimer points must have >= 10 training rows; warn and list the uncovered bins otherwise (training coverage below 6 A is not in the metadata) |

## Probe inputs

Most checks build their own probes from the canonical dimer records (fine scans, rotations, growing boxes, charge
states). Several take extra input, and say `not_evaluable` without it:

| flag | what it supplies | builder |
|---|---|---|
| `--ipea FILE` | vertical IP/EA probes with DFT references, for `charge_state` | `tools/priors_ipea.py` |
| `--sep-frames FILE` | separated and single-component frames, for `charge_offset_by_Q` | `tools/priors_sepframes.py` |
| `--seen FILE` | `seen_by` contamination flags, for `contamination` | `tools/seen_by.py` |
| `--train-coverage FILE` | training rows (gap distance × charge class), for `distribution_shift` | your own training metadata |
| `--baseline FILE` | an earlier priors JSON, for paired checks (e.g. IP/EA damage against the base model) | a previous run |
| `--ref-model SPEC` | a second backend for `precision` (fp32 vs fp64, CPU vs GPU) | — |
| `--moments FILE` | fragment DFT moments (`frags.json` of the dimer build, or `$LRBENCH_FRAG_MOMENTS`), for `orientation_far` and `orientation_adversarial` | the dimer build |

`tools/priors_table.py` turns several priors JSONs into one check × model table.

## How the suite is tested

The checks are validated the way the benchmark asks models to be validated: by planting defects and requiring that
each is caught, and that **nothing else fires**. `src/lrbench/priors/synthetic.py` holds a clean analytic
point-charge model and fourteen subclasses that each break exactly one prior — a plateau in the gap window, a global
charge-equilibration leak, no tail beyond the cutoff, a spurious long-range term between neutral fragments, a
separated-charge offset, charge blindness, non-gradient forces, an energy step with no force, a broken invariance,
per-call noise, a box-dependent energy, a shifted IP, a model that answers from a cache, and a Coulomb force that
reverses sign in some orientations only (`OrientPatch`, below).

`tests/test_priors.py` runs the clean model (everything must pass) and each defect (its own check must fail, the
others must not). If you add a check, add its planted defect too — a check with nothing to catch it has never been
shown to work.

One of those defects is worth singling out. `Caching` models a calculator that returns a previous structure's
numbers when a call differs only in total charge — a real bug, and one this repository had: ASE's `check_state`
compares geometry and ignores `atoms.info`, so until it was fixed, two charge states of one geometry could be scored
from a single evaluation. The fix (`lrbench.backends.drop_ase_cache`) now clears ASE's own result cache before every
structure, which means the planted defect has to cache *inside* the calculator, where the harness cannot reach it —
exactly where a real model's cache lives. `charge_conditioning` is the check that catches it.

## Orientations, and the tail of the distribution

The fine scans move fragment B along one axis in one orientation, and the checks that read them report medians.
Both choices can hide a defect.

An LR model with a link-state mixture shows how. The model is PET-OMol-S with a long-range head whose Coulomb term
is masked by a backbone link-state mixture.
- **The defect.** Under thermal sampling, its mean force on NH₄⁺···Gdm⁺ at 8.5 Å centre-of-mass separation has the
  wrong sign: −0.25 and −0.15 eV/Å for two training seeds, against Coulomb +0.20.
- **What it passed.** `conservativeness` (median 0.010), `seam`, `energy_force`, `nve_short`, and its own
  single-orientation scan (+0.19 eV/Å).
- **Where the error sits.** It is static and depends on orientation. A link joins the two cations in 68 % of rigid
  orientations at that separation, and about half of all orientations have the wrong sign.

Two checks sample the orientation instead.

- **`orientation_far`.** The sample:
  - each dimer pair, with both fragments rigid and each turned by its own uniformly random rotation about its centre
    of mass;
  - centre-of-mass separations r = 10, 12 and 15 Å (the **far** band, which is ruled) and 7.5–9.5 Å with closest
    contact ≥ 4 Å (the **mid** band, reported).

  It is scored against the multipole oracle (`oracle:multipole`). The oracle needs only the fragments' own DFT
  moments, so it gives a reference for any orientation without new DFT.

  The force on B along the axis is compared with the oracle's −dE/dr. Per class, the check reports:
  - the wrong-sign fraction;
  - the missing-tail fraction;
  - the median, q95 and max of ε_F = (F_model − F_oracle) / rms(F_oracle);
  - the orientation-dependent energy error.

  The sign is trusted only where the oracle's last two terms, the quadrupole and classical induction, cannot flip
  it.
- **`orientation_adversarial`.** It searches for a witness rather than sampling.
  - **Search.** Nelder–Mead over the rotation of each fragment and r, maximising |ε_F|. The starts are the worst
    sampled orientations plus random restarts.
  - **Output.** The worst geometry per pair and band goes to the JSON, and with `--orient-out` to extxyz, with the
    model and oracle forces in `info`.

**The mid band is not ruled.**
- **Why.** At 7.5–10 Å the oracle leaves out charge penetration, exchange and charge transfer, and it has not yet
  been validated against DFT there.
- **How it is reported.** A violation in the mid band is a `warn`. `dft_stage` lists the geometries a validation
  should start from: per pair, the largest model–oracle disagreements and the largest oracle truncation spreads.
- **Sign versus size.** For two charged fragments the monopole term dominates at these separations, so the sign of
  the force is much better determined than its size.

Both checks need `--moments frags.json`, the fragment moments of the dimer build. Without it they are
`not_evaluable`.

`--scan-orientations K` runs `seam`, `energy_force` and `conservativeness` on K extra random rigid orientations of
every pair as well as on orientation 0. These three checks need no reference at the scanned geometry.

Every median those checks report now has its q95 and max beside it:
- `conservativeness`: per region;
- `virial_tail`: |p − 1| per class;
- `seam`: the slope ratio |dE/dr| / |dE_Coul/dr| per class, in d = 6–12 Å.

These are reported, not ruled: the pass/fail rules are unchanged until protocol v1.

The planted defect is `OrientPatch`.
- **The defect.** The Coulomb term between a charged molecular cluster and its partner reverses sign when the
  molecule's first atom faces the partner.
- **The canonical orientation.** That atom faces away, so every scan-based check passes the model, and the test
  matrix asserts it. Only the orientation checks catch it.

## Reading a report

`--md` writes the table below; the JSON carries the same with full numbers. From a Lennard-Jones calculator scored
against the real dimer set — a model with no electrostatics at all, which is why it is a useful smoke test of the
suite:

```
| check                  | kind     | outcome  | result                                                        |
|------------------------|----------|----------|---------------------------------------------------------------|
| far_field_exponent     | physical | **FAIL** | ion_pair 0/36 (tail missing 36); ion_polar 0/20 (tail missing) |
| eps_far                | physical | **FAIL** | ion_pair eps -1.00; same_sign eps -1.00                        |
| seam                   | physical | pass     | no seam                                                        |
| size_consistency       | physical | **FAIL** | 4/6 pairs separable; worst Cl_Cl:o0 R = -144.0 meV             |
| invariance             | physical | pass     | invariant within tolerance                                     |
| charge_conditioning    | model    | **FAIL** | no response to Q for 6/6 probes                                |
| determinism            | model    | pass     | max |dE| 0 eV, max |dF| 0 eV/A                                 |
| offset_free_degeneracy | data     | **FAIL** | degenerate for ion_pair, ion_polar, same_sign                  |
| contamination          | data     | n/a      | no seen_by flags (--seen)                                      |
```

Note the last two rows, which are not about the model. `offset_free_degeneracy` fails because for a model this flat
the offset-free error at 8 Å cannot distinguish it from a line — so that metric must not be quoted for it, and the
check says which one to quote instead. `contamination` is `n/a`, not `pass`, because no `seen_by` file was given.

## Related work: reference-free suites

Two recent suites test potentials without reference data, and both are worth reading alongside this one.

[**Reference-free certification**](https://arxiv.org/abs/2610.00585) (arXiv:2610.00585) defines 14 probes over 64
pretrained potentials: symmetry and invariance, self-consistency and integrability, statistical-mechanical
equilibrium, and regularity. Each probe scores `S = exp(-d/lambda)` and the certificate is their **geometric mean**,
chosen so that a certificate "cannot hide a single failing axis behind many passed gates" — the same construction
and the same argument as this project's long-range score, arrived at independently. Its fine-tuning series is also
direct evidence for the premise here: the naively fine-tuned model has the *lowest* held-out force error, 0.055
eV/A against 0.060 and 0.151, and is the only one whose certificate collapses, to 0.04; in MD it reaches nearly
2,000 K within 100 steps and returns non-finite coordinates at step 140, where the others run 10,000 steps.

[**MLIP Arena**](https://arxiv.org/abs/2509.20630) (arXiv:2509.20630) scores the *shape* of curves rather than
agreement with DFT: diatomic potential-energy curves with force-gradient conservation, tortuosity and sign flips;
equation-of-state monotonicity; MD stability; NVE energy conservation.

**Neither suite contains a long-range test.** Across their 23 tests there is nothing on electrostatics, total
charge, Coulomb tails, polarisation, or the energy of two fragments as they separate beyond a cutoff. Both say
why: certification's substrates are "small and elemental by design" (13-atom clusters, a trimer, a four-atom
cell), and MLIP Arena's diatomic curves stop at 3.1 r_vdW, while its one dispersion-adjacent task adds D3(BJ) to
every model and so measures MLIP + D3 rather than the model's own far field.

`conservativeness` and `virial_tail` come from reading them. The first is their integrability family, restricted to
the one geometry these sets already contain and reported separately near and far. The second is the idea their
substrates cannot reach: a virial identity is a statement about the whole potential, so moving it onto a charged
pair at 12 A and beyond turns a model-agnostic probe into a long-range diagnostic, and one that needs no reference
at all. Measured: a model with an explicit Coulomb term recovers p = 0.996 on like-charged pairs, while two purely
local models return *not evaluable* — they have no tail to form the ratio from.

Their trimer probe overlaps task **D1**: the same three-body term, required to fall monotonically to zero by 12 A.
Theirs adds a total-variation penalty for a non-monotone approach, which D1 does not have. Adding it needs a denser
apex sweep than the current `trimers` set carries (three distinct distances, 9/11/14 A), so it is a data change
rather than a code change.

## Known limitations

- The `ext_*` reference-only sets do not trigger the level-of-theory refusal, because set levels are inferred from
  records that carry an `energy` (see [datasets.md](datasets.md)). Pass `--allow-level-mismatch` and state the
  mismatch; `level_of_theory` reports `not_evaluable` for a model whose own level is unknown.
- `nve_short` and `energy_force` use finite differences and short trajectories; on a stiff potential they can fail
  for reasons of step size rather than physics. The report gives the worst case and where it occurred so the two can
  be told apart.
- `distribution_shift` and `contamination` depend entirely on metadata you supply about the training set. Without it
  they are `n/a`, which is an honest "unknown", not a clean bill of health.
