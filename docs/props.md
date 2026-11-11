# Electrostatic properties

Interaction energies tell you whether the long-range *energy* is right. The `props` set asks whether the underlying
*electrostatics* are right: dipoles, quadrupoles, polarisabilities and the induced response. A model can get a tail
roughly right with unphysical charges, and that shows up here.

`props` is **not** part of `--sets all` (its references are built separately); name it explicitly.

## References

ωB97M-V/def2-TZVPD (ORCA 6.1.1, OMol25 SCF settings, `%elprop Dipole Quadrupole Polar 1`, analytic CP-SCF α) for:

- a stratified OMol25 `val_sel_clean` subset — 50 neutral, 50 cation, 50 anion, 50 open-shell, ≤ 40 atoms;
- the 24 dimer monomers;
- ion+polar and ion+nonpolar dimer scans (orientation 0) with their DFT Löwdin charges.

```sh
python tools/props_build.py CALCDIR <canonical-dir> DATA_DIR        # build props.jsonl from ORCA output
lrbench eval --model model.gguf --sets props --data DATA_DIR --out model.props.json
python tools/props_eval.py --model "ase:lrbench.adapters:mace_polar?model='polar-1-m'" --alpha field \
    --data DATA_DIR --out polar.props.json                           # + polarisability by finite field
python tools/props_eval.py --oracle lowdin --data DATA_DIR --out lowdin.props.json   # charge-only floor
python tools/props_summary.py DATA_DIR frags.json props_summary results/*.props.json
```

## Conventions

- μ and Θ (traceless Buckingham, 3/2·M − 1/2·tr M) are about the **centre of mass**, for model and reference alike.
  For ions they depend on that origin, so the convention is not optional. Atomic units.
- Model moments are computed from the per-atom charges the backend returns. For LES-type models those are *latent*
  charges, not trained to be physical: they are scored as given, and a zero-intercept slope reports their scale
  rather than pretending the magnitude is meaningful. A model's own dipole output is used for neutral molecules only.
- α = −d²E/dF² by central differences in a uniform field for models that accept one; Σ ratio_i · α_free(Z_i) for
  SO3LR (isotropic, no coupling); otherwise N/A — not zero, N/A.
- Δμ(d): the dipole change of the neutral fragment relative to the scan's far point, from per-atom charges, against
  DFT Löwdin charges. Charge-only on both sides, so out-of-plane polarisation of a planar molecule is invisible to
  both — a limitation of the comparison, not a model error.

## Metrics

MAE and RMSE with 95 % bootstrap over molecules, by charge state: |μ| and vector errors, Θ component and norm errors,
α isotropic and anisotropy errors.

## Vertical IP / EA

```sh
python tools/props_ipea.py IPEA_CALCDIR ipea.json
```

Over ORCA chains for N, N−1 and N+1 electrons at fixed geometry (~240 fragments): IP, EA, χ = (IP + EA)/2,
J = IP − EA, HOMO/LUMO of N, ⟨S²⟩. Anions whose N+1 HOMO is above zero are basis-bound; they are **kept and
flagged** rather than dropped. These are the quantities a charge-equilibration model implicitly assumes, so they are
the right prior to check before blaming a model's charges.
