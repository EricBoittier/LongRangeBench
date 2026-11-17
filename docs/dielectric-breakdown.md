# Dielectric-breakdown toggle

Both notebooks (`notebooks/model_scorecard.py` and `notebooks/long_range_diagnostics.py`) carry a dielectric-breakdown
toggle next to the multivalent switch. A frame is labelled *breakdown* when charge has moved across a boundary that a
polarisable medium should hold.

| setting | values | default |
|---|---|---|
| `breakdown` | `include` (keep and flag), `exclude` (drop from the per-frame recomputed scores and counts, show apart), `only` (keep the flagged frames alone) | `include` |
| `breakdown_rule` | `charge`, `physics` | `physics` |
| physics threshold | largest \|DFT Löwdin fragment sum − formal charge\| over the fragments | 0.1 e |

Set them in the notebook widgets, or headless with `LRB_BREAKDOWN`, `LRB_BREAKDOWN_RULE` and `LRB_BREAKDOWN_THR`
(the scorecard also takes `--breakdown`, `--breakdown-rule` and `--breakdown-thr` after `--` in `marimo export`).
Every breakdown-aware score is labelled with the mode and the rule it used. The toggle reaches per-frame recomputed
scores (the scorecard's dimer E_int MAE) and the frame counts; committed result rows and precomputed program scores
include every frame whatever the state, and their panels say so.

**Rule `charge`.** A frame is flagged when its electrode charge is q ≥ +1 e. De Kam, Rinke, Stein & Loche exclude
their q = +1 e electrode training data because those data are prone to dielectric breakdown. The rule reads the
per-record fields `electrode_charge` (e) and `surface_charge` (µC/cm²) when a set carries them; a |σ| threshold is
available as `BD_SIGMA_THR` in `notebooks/lrb_data.py` and is off by default. The current sets have no electrodes, so
this rule reports "no electrode frames in this set" and changes no score. The fields are kept so that an
electrode–water set (for example Pt–water) works without code changes.

**Rule `physics`.** Bergmann, Reuter & Hörmann show that PBE water is polarisable only inside its band-edge window
(φ ≈ 2–7 V). Beyond that window, charge transfers between the metal and the water. φ(q) then flattens, ∂φ/∂q
collapses, and the water's Hirshfeld net charge shifts. Their filter accepts (∂φ/∂q)⁻¹ ≈ C₀ = 2.124 ± 1.5 µF/cm². On
molecular sets, the analogue is fractional charge transfer between fragments. A frame is flagged when any
fragment's DFT Löwdin sum (Hirshfeld where a set has no Löwdin sums) differs from its formal charge by more than the
threshold. Frames without fragment charge labels (single-fragment frames, sets without population analysis) are
counted as "no charge labels" and are not flagged. Where a set carries response labels (`phi` in V, `capacitance` in
µF/cm²), frames with φ outside 2–7 V or a capacitance outside C₀ ± 1.5 µF/cm² are flagged too. No current set has
these labels, so the notebooks print "no labels in this set" for them; this part of the rule is untested.

**Relation to CT-ambiguous.** The breakdown flag and the CT-ambiguous class (`ion_pair_ct_ambiguous`,
`audit:ct_ambiguous`) are separate labels. CT-ambiguous is shown in its own column and never enters the breakdown
flag, so `exclude` keeps every CT-ambiguous frame that the rule does not flag. On the dimers set, at 0.1 e, the
physics rule flags 69 of 1,254 E_int groups. 25 of them are among the 60 CT-ambiguous ion pairs; the other 44 are
not CT-ambiguous. The remaining 35 CT-ambiguous pairs are not flagged.

**Why the default is include.** The benchmark scores how well a model emulates DFT, and DFT's pathologies are part of
what it emulates. Breakdown frames therefore stay in every score by default and are flagged. `exclude` and `only`
exist to show how much of a score these frames carry.

**Data.** The builder (`tools/lrp_scorecard.py`, `breakdown_fields`) stores only the raw labels per frame:
`dq_frag`, plus the electrode fields when any record has them. The notebook applies the rule, the threshold
and the mode, as it does with `maxq` for the multivalent switch.

## References

- de Kam, Rinke, Stein & Loche, LOREM (electrified interfaces), preprint (2026).
- Bergmann, Reuter & Hörmann, *J. Chem. Phys.* **164**, 174110 (2026).
