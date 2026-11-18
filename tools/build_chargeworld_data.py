"""Build the ChargeWorld notebook data folder (notebooks/data/chargeworld/) from the ChargeWorld sources.

    python tools/build_chargeworld_data.py                 # everything (about 2 min; needs the ChargeWorld package)
    python tools/build_chargeworld_data.py --results-only  # re-read the run folders only (what the notebook's
                                                           # "rebuild from sources" button runs)

Sources (read only):
  CW_ROOT  default ~/data/lr-charge-world      generator package `cw`, data/<tier>/npz, baselines, configs
  CW_RUN   default ~/data/lr-charge-world-run  training runs (runs/, collapse/runs/, noise_ladder/runs/),
                                                species-charge toy (gradconf/out), noise variants (data/T1_x2_*)

Only whitelisted fields are written: no model paths, no host names, no job identifiers. Large arrays are reduced:
  * structures.json: 6-8 test structures per tier, coordinates rounded to 1 mA, charges to 1e-4 e;
  * toy.npz: the species-charge toy's quadratic forms in the symmetric 78-term basis (fp64 sums for full-batch
    training, fp32 per-structure upper triangles for minibatch training and the gradient-conflict analysis);
  * traces.json: probe traces every 10 steps (all logged steps), per-species fields only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "notebooks" / "data" / "chargeworld"
CW_ROOT = Path(os.environ.get("CW_ROOT", "~/data/lr-charge-world")).expanduser()
CW_RUN = Path(os.environ.get("CW_RUN", "~/data/lr-charge-world-run")).expanduser()

TIERS = [  # name, what it adds, what it nests to (DESIGN.md section 4, shortened)
    ("T0", "±1 e point charges by species (2 species), LJ, periodic Ewald; 128-atom boxes", "the LES random-charge box"),
    ("T1", "12 charge types (±0.5, ±1 hard and soft, ±2, ±3, two neutral spacers), total charge Q from -2 to 2",
     "root of the nested ladder"),
    ("T1s", "T1 charges as Gaussian densities (width 0.6-1.4 Å)", "T1 at width 0"),
    ("T2", "charge equilibration (QEq): hardness 20-4,000 eV, soft atoms move 0.06-0.3 e", "T1s at infinite hardness"),
    ("T3a", "non-local charge transfer between two clusters 3-25 Å apart (open boundary)", "T2 physics"),
    ("T3b", "induced dipoles solved jointly with the charges", "T2 at zero polarisability"),
    ("T3c", "screened (Yukawa) kernel, λ = 3 Å", "T2 at infinite λ"),
    ("T3d", "boundary conditions: periodic, open droplets and slabs, one third each", "T2 for the periodic third"),
    ("T4a", "white label noise σ_F 50 meV/Å, σ_E 2 meV·√N", "T2 at zero noise"),
    ("TS", "short-range chemistry (Morse wells and a three-body term) on T2", "T2 at zero weight"),
    ("T5", "QEq + induction + short-range chemistry + noise, jitter, outliers and SCF-like charge error", "-"),
]
T4_ALL = {"T4a": "white noise σ_F 50 meV/Å", "T4b": "heteroscedastic noise σ_F 10-200 meV/Å",
          "T4c": "position jitter 0.005 Å", "T4d": "outliers (2 % of structures, 1 % of atoms)",
          "T4e": "missing labels (30 % energy-only, 30 % force-only)", "T4f": "cutoff artifact (shifted-force Coulomb, 8 Å)",
          "T4g": "SCF-like charge error"}
EXPLORE = ["T0", "T1", "T1s", "T2", "T3a", "T3b", "T3c", "T3d", "T4a", "TS", "T5"]
BASE_KEYS = ["exact", "point_true_q", "fixed_species_q", "fixed_species_q_tier_kernel", "local2b_rc3", "local2b_rc6"]


def jdump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    print(f"  wrote {name}: {(OUT / name).stat().st_size / 1e3:.0f} kB")


def r(x, nd):
    return np.round(np.asarray(x, float), nd).tolist()


def cw():
    sys.path.insert(0, str(CW_ROOT))
    from cw.io import load_npz  # noqa: E402
    from cw.electro import Elec  # noqa: E402
    return load_npz, Elec


def species_table():
    S = json.loads((CW_ROOT / "configs" / "species12.json").read_text())["species"]
    return [dict(name=s["name"], Z=int(s["Z"]), q0=float(s["q0"]), eta=float(s["eta"]), sigma=float(s["sigma"]),
                 alpha=float(s["alpha"])) for s in S]


# ------------------------------------------------------------------------------------------------ tiers, baselines
def build_tiers():
    base = json.loads((CW_ROOT / "baselines" / "report.json").read_text())
    rows = []
    for name, adds, nests in TIERS + [(k, v, "T2 at zero noise") for k, v in T4_ALL.items() if k != "T4a"]:
        man = CW_ROOT / "data" / name / "MANIFEST.json"
        splits = {k: v["structures"] for k, v in json.loads(man.read_text())["splits"].items()} if man.exists() else {}
        b = base.get(name, {})
        bl = {}
        for k in BASE_KEYS:
            t = b.get(k, {}).get("test")
            if t:
                bl[k] = {kk: round(float(v), 4) for kk, v in t.items()}
        rows.append(dict(tier=name, adds=adds, nests=nests, splits=splits, baselines=bl,
                         noise_floor_F=round(float(b["analytic_floor_F_meV_A"]), 2) if "analytic_floor_F_meV_A" in b else None))
    jdump("tiers.json", dict(tiers=rows, species=species_table()))


# ------------------------------------------------------------------------------------------------ structures
def pick_T1(load_npz):
    """first 4 test structures plus the first 4 that the trapped Phase-1 seed 15 gets mixed (if its predictions exist)."""
    d = load_npz("T1", "test"); na = d["na"]; qt = d["q"].astype(float); Z = d["Z"]
    # first four structures with at least three charged species (four T1 test boxes hold only spacers)
    pick = [k for k in range(len(na) - 1) if len(np.unique(Z[na[k]:na[k + 1]][np.abs(qt[na[k]:na[k + 1]]) > 0.25])) >= 3][:4]
    pred = CW_RUN / "runs" / "T1_A1_mid_s15" / "pred_N16.json"
    if pred.exists():
        P = json.loads(pred.read_text())["frames"]
        for k, p in enumerate(P):
            q = np.asarray(p["latent_charges"], float).reshape(na[k + 1] - na[k], -1)[:, 0]; t = qt[na[k]:na[k + 1]]
            g = 1.0 if q @ t >= 0 else -1.0; m = np.abs(t) > 0.25; z = Z[na[k]:na[k + 1]]
            mixed = any(((g * q * t < 0) & m & (z == s)).sum() > (m & (z == s)).sum() / 2 for s in np.unique(z[m]))
            if mixed and k not in pick:
                pick.append(k)
            if len(pick) >= 8:
                break
    return pick


def quad_one(Elec, R, cell, sp, nsp):
    """per-species quadratic forms of the point-charge Coulomb energy and forces (as gradconf/scripts/build_quad.py)."""
    el = Elec(R, cell, "periodic", 0.0)
    J = el.Jqq("point"); S = np.zeros((len(sp), nsp)); S[np.arange(len(sp)), sp] = 1.0
    A = S.T @ J @ S; one = np.ones(len(sp)); E1 = el.forces(one)["coul_point"]
    H = np.zeros((len(sp), 3, nsp))
    for t in range(nsp):
        m = sp == t
        if m.any():
            F = el.forces(one + m)["coul_point"]; H[:, :, t] = np.where(m[:, None], F / 2.0 - E1, F - E1)
    return A, H


def build_structures():
    load_npz, Elec = cw()
    spec = {s["Z"]: s for s in species_table()}
    out = {}
    for tier in EXPLORE:
        d = load_npz(tier, "test"); na = d["na"]
        pick = pick_T1(load_npz) if tier == "T1" else list(range(min(6, len(na) - 1)))
        rows = []
        for k in pick:
            a, b = int(na[k]), int(na[k + 1])
            row = dict(index=int(k), Z=d["Z"][a:b].astype(int).tolist(), R=r(d["R"][a:b], 3),
                       cell=r(d["cell"][k], 3), bc=int(d["bc"][k]), Q=int(d["Q"][k]), q=r(d["q"][a:b], 4),
                       F=r(d["F_clean"][a:b] if "F_clean" in d else d["F"][a:b], 3),
                       E=round(float(d["E_clean"][k]), 4), frag=d["frag"][a:b].astype(int).tolist(),
                       terms={t: round(float(d[t][k]), 4) for t in ("E_onsite", "E_coul_point", "E_coul_smear", "E_ind",
                                                                       "E_lj", "E_sr_chem") if t in d})
            if "mu" in d and np.abs(d["mu"][a:b]).max() > 1e-6:
                row["mu"] = r(d["mu"][a:b], 4)
            if tier == "T1":   # exact quadratic forms for the sign-flip demonstration
                Zs = sorted(spec) if tier != "T0" else None
                zi = {z: i for i, z in enumerate(Zs)}
                sp = np.array([zi[z] for z in row["Z"]])
                A, H = quad_one(Elec, d["R"][a:b].astype(float), d["cell"][k].reshape(3, 3).astype(float), sp, len(Zs))
                c = np.array([spec[z]["q0"] for z in Zs])
                Fp = c[sp][:, None] * np.einsum("iaj,j->ia", H, c)
                assert abs(0.5 * c @ A @ c - d["E_coul_point"][k]) < 1e-3, "quad energy check"
                assert np.abs(Fp - d["F_point"][a:b]).max() < 1e-3, "quad force check"
                row["quad"] = dict(A=r(A, 6), H=np.round(H, 6).reshape(len(sp), -1).tolist(), sp=sp.tolist())
            rows.append(row)
        out[tier] = rows
        print(f"  {tier}: {len(rows)} structures")
    out["_species_Z"] = sorted(spec)
    jdump("structures.json", out)


# ------------------------------------------------------------------------------------------------ distributions
def hist(x, bins):
    h, e = np.histogram(x, bins=bins); return dict(edges=r(e, 4), counts=h.tolist())


def build_distributions():
    load_npz, _ = cw()
    spec = species_table(); out = {}
    for tier in EXPLORE + ["T1_combo", "T1_Qood"]:
        t, split = (tier.split("_") + ["test"])[:2] if "_" in tier else (tier, "test")
        d = load_npz(t, split); na = d["na"]; N = np.diff(na)
        q = d["q"].astype(float); Z = d["Z"].astype(int)
        F = d["F_clean"] if "F_clean" in d else d["F"]
        cell = d["cell"].reshape(-1, 3, 3); L = np.abs(np.linalg.det(cell)) ** (1 / 3)
        per = {}
        for z in np.unique(Z):
            qq = q[Z == z]; per[int(z)] = dict(n=int(qq.size), mean=round(float(qq.mean()), 4), sd=round(float(qq.std()), 4),
                                               p05=round(float(np.quantile(qq, 0.05)), 4), p95=round(float(np.quantile(qq, 0.95)), 4))
        out[tier] = dict(n=int(len(N)), atoms=hist(N, 20), E_atom=hist(d["E_clean"] / N, 30),
                         Fnorm=hist(np.linalg.norm(F, axis=1), np.linspace(0, float(np.quantile(np.linalg.norm(F, axis=1), 0.995)), 41)),
                         q=hist(q, np.linspace(-3.5, 3.5, 71)), Q=dict(zip(*[v.tolist() for v in np.unique(d["Q"], return_counts=True)])),
                         L=hist(L[d["bc"] == 1], 20) if (d["bc"] == 1).any() else None, species=per)
        out[tier]["Q"] = {str(k): v for k, v in out[tier]["Q"].items()}
    # species co-occurrence (structures in which both species appear): T1 train, val, test, combo
    Zs = [s["Z"] for s in spec]; zi = {z: i for i, z in enumerate(sorted(Zs))}
    co = {}
    for split in ("train", "val", "test", "combo"):
        d = load_npz("T1", split); na = d["na"]; C = np.zeros((12, 12), int)
        for k in range(len(na) - 1):
            pres = sorted({zi[int(z)] for z in d["Z"][na[k]:na[k + 1]]})
            for i in pres:
                for j in pres:
                    C[i, j] += 1
        co[split] = C.tolist()
    out["_cooc"] = dict(Z=sorted(Zs), counts=co)
    jdump("distributions.json", out)


# ------------------------------------------------------------------------------------------------ run results
_test = {}


def test_set(tier):
    if tier not in _test:
        load_npz, _ = cw(); d = load_npz(tier, "test")
        _test[tier] = (d["na"], d["q"].astype(float), d["Z"].astype(int), d["F"].astype(float))
    return _test[tier]


def score(run_dir: Path, tier: str, keep_q: list[int] | None = None):
    """pre-registered failure rule (one global gauge per run; fail if a species with |t| > 0.25 e has more than half
    of its atoms at the wrong sign, or the gauge-corrected charge RMSE is above 0.5 e) plus secondary readouts."""
    na, qt, Z, F = test_set(tier); N = np.diff(na)
    P = json.loads((run_dir / "pred_N16.json").read_text())
    P = P["frames"] if isinstance(P, dict) else P
    assert len(P) == len(N), (run_dir.name, len(P), len(N))
    if "latent_charges" not in P[0]:          # no charge head (plain PET): force and energy errors only
        fe = [np.mean(np.abs(np.asarray(p["forces"], float).reshape(-1, 3) - F[na[k]:na[k + 1]])) * 1e3 for k, p in enumerate(P)]
        return dict(fail=None, flipped=[], q_rmse=None, gauge=None, mixed_frac=None, n_mixed=None, F_mae=round(float(np.mean(fe)), 2),
                    F_mae_mixed=None, F_mae_coherent=None, struct_gauge_plus=None, struct_gauge_minus=None, species={})
    qs, fe, gs, mixed, Fp = [], [], [], [], []
    for k, p in enumerate(P):
        assert int(p["n_atoms"]) == N[k], "prediction frame order does not match the test split"
        q = np.asarray(p["latent_charges"], float).reshape(N[k], -1)[:, 0]; t = qt[na[k]:na[k + 1]]
        g = 1.0 if q @ t >= 0 else -1.0; m = np.abs(t) > 0.25; bad = (g * q * t < 0) & m
        # sign-mixed structure: under its own best gauge, more than 1 % of its charged atoms (|t| > 0.25 e) have the
        # wrong sign (the Phase-1 definition behind the 262 / 49 meV/A force split)
        mixed.append(bool(m.any() and np.mean(bad[m]) > 0.01))
        f = np.asarray(p["forces"], float).reshape(-1, 3)
        fe.append(np.mean(np.abs(f - F[na[k]:na[k + 1]])) * 1e3); qs.append(q); gs.append(g); Fp.append(f)
    qall = np.concatenate(qs); G = 1.0 if qall @ qt >= 0 else -1.0
    rmse = float(np.sqrt(np.mean((G * qall - qt) ** 2)))
    m = np.abs(qt) > 0.25
    frac = {int(s): float(np.mean(G * qall[m & (Z == s)] * qt[m & (Z == s)] < 0)) for s in np.unique(Z[m])}
    per = {}
    for s in np.unique(Z):
        sel = Z == s
        per[int(s)] = dict(mean=round(float((G * qall[sel]).mean()), 4), absmean=round(float(np.abs(qall[sel]).mean()), 4),
                           sd=round(float(qall[sel].std()), 4), truth=round(float(qt[sel].mean()), 4),
                           wrong=round(frac.get(int(s), 0.0), 4))
    fe = np.array(fe); mixed = np.array(mixed); gs = np.array(gs)
    res = dict(fail=bool(any(v > 0.5 for v in frac.values()) or rmse > 0.5), flipped=[s for s, v in frac.items() if v > 0.5],
               q_rmse=round(rmse, 4), gauge=G, mixed_frac=round(float(mixed.mean()), 4), n_mixed=int(mixed.sum()),
               F_mae=round(float(fe.mean()), 2),
               F_mae_mixed=round(float(fe[mixed].mean()), 1) if mixed.any() else None,
               F_mae_coherent=round(float(fe[~mixed].mean()), 1) if (~mixed).any() else None,
               struct_gauge_plus=int((gs > 0).sum()), struct_gauge_minus=int((gs < 0).sum()), species=per)
    if keep_q:
        res["q_struct"] = {str(k): r(qs[k], 4) for k in keep_q if k < len(qs)}
        res["F_struct"] = {str(k): r(Fp[k], 3) for k in keep_q if k < len(Fp)}
    return res


def val_curve(run_dir: Path):
    f = run_dir / "log.jsonl"
    if not f.exists():
        return None
    rows = {}
    for line in f.read_text().splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_f_rmse_mev_a" in j:
            rows[int(j.get("epoch", -1))] = (round(j["val_f_rmse_mev_a"], 2), round(j.get("val_e_rmse_mev_atom", np.nan), 3))
    ks = sorted(rows)
    return dict(epoch=ks, F_rmse=[rows[k][0] for k in ks], E_rmse=[rows[k][1] for k in ks])


def probe_trace(run_dir: Path):
    f = run_dir / "charge_diag.jsonl"
    if not f.exists():
        return None
    recs = {}
    for line in f.read_text().splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs[j["step"]] = j     # a resumed run repeats steps: keep the last record
    st = sorted(recs)
    sp = sorted({int(s) for k in st for s in recs[k]["probe"]["species"]})
    out = dict(step=st, n_mixed=[recs[k]["probe"]["n_mixed"] for k in st],
               rmse_sign=[round(recs[k]["probe"]["rmse_sign"], 4) for k in st],
               loss_f=[round(recs[k].get("loss_f", np.nan), 5) if "loss_f" in recs[k] else None for k in st],
               loss_q=[round(recs[k].get("loss_q", 0.0), 5) if "loss_q" in recs[k] else None for k in st],
               agree={}, mean={})
    for s in sp:
        out["agree"][str(s)] = [round(recs[k]["probe"]["species"].get(str(s), {}).get("agree_g", np.nan), 3) for k in st]
        out["mean"][str(s)] = [round(recs[k]["probe"]["species"].get(str(s), {}).get("mean", np.nan), 4) for k in st]
    # relative-sign pattern per charged species under the probe's own global sign, and the sign-choice step (the step
    # after which the pattern equals the final one), as in the replicate set's analysis script
    pats = []
    for k in st:
        sp_ = {z: v for z, v in recs[k]["probe"]["species"].items() if abs(v["truth"]) > 0.25}
        G = 1 if sum(v["n"] * v["mean"] * v["truth"] for v in sp_.values()) >= 0 else -1
        pats.append({z: (1 if (v["agree"] > 0.5) == (G > 0) else -1) for z, v in sp_.items()})
    out["charged"] = sorted(int(z) for z in pats[-1])
    out["n_wrong"] = [sum(v < 0 for v in p.values()) for p in pats]
    j = len(pats) - 1
    while j > 0 and pats[j - 1] == pats[-1]:
        j -= 1
    out["choice_step"] = int(st[j])
    return out


def clean_nan(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(v) for v in o]
    return o


SCHEME = {"S0": "plain LES", "S1": "charge-correlation anchor", "S2": "charge-MSE anchor", "S4": "EMA 0.999"}
LADDER = {"EF0.5": "E/F noise 0.5×", "EF1": "E/F noise 1×", "EF2": "E/F noise 2×", "W25": "white noise 25×",
          "P5": "pathology tail 5×", "B4E1": "batch 4 in epoch 1", "B64": "batch 64", "WU10": "warm-up 10 %",
          "COMBO": "held-out pairs co-occur", "B4": "batch 4 throughout"}


def build_results():
    structs = json.loads((OUT / "structures.json").read_text()) if (OUT / "structures.json").exists() else {}
    keep = {t: [s["index"] for s in structs.get(t, [])] for t in ("T0", "T1", "T2")}
    runs, traces = [], {}
    # Phase 1: architecture comparison (A0 plain PET, A1 PET + LES charge head), seeds 13-16
    for d in sorted((CW_RUN / "runs").glob("T*_A*_s*")):
        tier, arch, cap, seed = d.name.split("_")
        if not (d / "pred_N16.json").exists():
            continue
        s = score(d, tier, keep.get(tier) if arch == "A1" else None)
        splits = {}   # the ChargeWorld scorer's per-split numbers (test, combo, Qood), for the pre-registered limits
        for sp_, f in (("test", "score_N16.json"), ("combo", "score_combo_N16.json"), ("Qood", "score_Qood_N16.json")):
            if (d / f).exists():
                j = json.loads((d / f).read_text())
                splits[sp_] = dict(F_mae=round(j["F_mae_meV_A_given"], 2),
                                   q_err_by_Z={k: round(v, 4) for k, v in j.get("q_err_by_Z", {}).items()})
        runs.append(dict(set="phase1", tier=tier, arm=f"{arch} {cap}", scheme="-", seed=int(seed[1:]), **s, curve=val_curve(d),
                         splits=splits))
    # replicate set with probe traces: plain LES, two anchors, EMA (seeds 13-20 as they finish)
    for d in sorted((CW_RUN / "collapse" / "runs").glob("T*_S*_s*")):
        if not ((d / "DONE").exists() and (d / "pred_N16.json").exists()):
            continue
        tier, sch, seed = d.name.split("_")
        s = score(d, tier, keep.get(tier) if tier == "T1" else None)
        runs.append(dict(set="anchors", tier=tier, arm="A1 mid", scheme=sch, seed=int(seed[1:]), **s, curve=val_curve(d)))
        traces[f"anchors/{d.name}"] = probe_trace(d)
    # noise ladder and smoothing arms (empty until the runs are copied back)
    for d in sorted(glob.glob(str(CW_RUN / "noise_ladder" / "runs" / "T1_*_s*")) + glob.glob(str(CW_RUN / "noise_ladder" / "*" / "runs" / "T1_*_s*"))):
        d = Path(d)
        if not ((d / "DONE").exists() and (d / "pred_N16.json").exists()):
            continue
        tier, arm, seed = d.name.split("_")
        s = score(d, tier)
        runs.append(dict(set="ladder", tier=tier, arm=arm, scheme=LADDER.get(arm, arm), seed=int(seed[1:]), **s, curve=val_curve(d)))
        traces[f"ladder/{d.name}"] = probe_trace(d)
    ref_sum = CW_RUN / "collapse" / "a46_summary.json"
    if ref_sum.exists():
        cs = {r["run"]: r.get("choice_step") for r in json.loads(ref_sum.read_text())["runs"]}
        for k, t in traces.items():
            name = k.split("/")[-1]
            if k.startswith("anchors/") and t and cs.get(name) is not None:
                assert t["choice_step"] == cs[name], (name, t["choice_step"], cs[name])
    # consistency with the hand-made Phase-1 analysis (PLAN_DELIVERABLES section 1)
    ref = {15: (262, 49), 16: (236, 66)}
    for x in runs:
        if x["set"] == "phase1" and x["tier"] == "T1" and x["arm"] == "A1 mid" and x["seed"] in ref:
            mm, cc = ref[x["seed"]]
            assert abs(x["F_mae_mixed"] - mm) < 1.5 and abs(x["F_mae_coherent"] - cc) < 1.5, (x["seed"], x["F_mae_mixed"], x["F_mae_coherent"])
    # summaries of the OOD splits (combo, Qood) for A0 vs A1
    ood = {}
    for f in sorted(CW_RUN.glob("summary_T*_N16.json")):
        j = json.loads(f.read_text()); key = f.stem.replace("summary_", "").replace("_N16", "")
        ood[key] = {arm: {k: v["vals"] for k, v in blk.items() if isinstance(v, dict) and "vals" in v}
                    for arm, blk in j.items() if isinstance(blk, dict) and "seeds" in blk}
    jdump("runs.json", clean_nan(dict(runs=runs, ood=ood, scheme_names=SCHEME, ladder_names=LADDER)))
    jdump("traces.json", clean_nan(traces))


# ------------------------------------------------------------------------------------------------ toy
def sym_basis():
    iu = np.triu_indices(12)
    P = np.zeros((144, 78))
    for k, (i, j) in enumerate(zip(*iu)):
        P[i * 12 + j, k] = 1.0
        if i != j:
            P[j * 12 + i, k] = 1.0
    return P


def toy_split(split):
    z = np.load(CW_RUN / "gradconf" / "out" / f"quad_T1_{split}.npz")
    A, H, sp, na, q0 = z["A"], z["H"], z["sp"], z["na"], z["q0"]
    n = len(na) - 1; N = np.diff(na).astype(float); sid = np.repeat(np.arange(n), np.diff(na))
    E = 0.5 * np.einsum("i,nij,j->n", q0, A, q0); F = q0[sp][:, None] * np.einsum("iaj,j->ia", H, q0)
    M = len(sp); K = np.zeros((M, 3, 12, 12)); K[np.arange(M), :, sp, :] = H; K = 0.5 * (K + K.transpose(0, 1, 3, 2))
    k = K.reshape(M, 3, 144); P = sym_basis(); iu = np.triu_indices(78)
    Mn = np.zeros((n, 78, 78)); bn = np.zeros((n, 78)); y2 = np.zeros(n)
    for i in range(n):
        a, b = na[i], na[i + 1]; kk = k[a:b].reshape(-1, 144) @ P; yy = F[a:b].reshape(-1); w = 1.0 / (3 * N[i])
        Mn[i] = kk.T @ kk * w; bn[i] = yy @ kk * w; y2[i] = (yy ** 2).sum() * w
    C = np.zeros((n, 12), int); np.add.at(C, (sid, sp), 1)
    return dict(A=A, E=E, N=N, Msum=Mn.sum(0), bsum=bn.sum(0), y2sum=y2.sum(), Mu=Mn[:, iu[0], iu[1]].astype(np.float32),
                b=bn, y2=y2, counts=C, q0=q0)


def build_toy():
    v = toy_split("val"); c = toy_split("combo")
    np.savez_compressed(OUT / "toy.npz", q0=v["q0"],
                        **{f"{p}_{k}": d[k] for p, d in (("val", v), ("combo", c))
                           for k in ("A", "E", "N", "Msum", "bsum", "y2sum", "Mu", "b", "y2", "counts")})
    print(f"  wrote toy.npz: {(OUT / 'toy.npz').stat().st_size / 1e6:.2f} MB")
    o = CW_RUN / "gradconf" / "out"
    h = json.loads((o / "hess.json").read_text())
    keep = ["eig_c0", "eig_q0", "eig_trap", "cond_q0", "cond_trap", "trap_min_eig", "n_unstable_c0",
            "overlap_true_with_unstable", "overlap_trap_with_unstable", "most_unstable_vs_true", "most_unstable_vs_trap",
            "most_unstable_vec", "trap_c", "q0", "loss_trap", "traj", "conflict_ratio", "frac_struct_pull_unflip",
            "proj_per_struct", "has_flipped_and_unflipped", "frac_pairs_negative_cos", "median_pair_cos"]
    arms = {}
    for f in sorted(o.glob("arms_*.json")):
        j = json.loads(f.read_text())
        arms[j["arm"]] = dict(trapped=j["trapped"], flips=j["flips"], rmse=r(j["rmse"], 5), commit=j["commit"],
                              loss_test=r(j["loss_test"], 6))
    jdump("toy_ref.json", dict(hess={k: h[k] for k in keep}, arms=arms))


# ------------------------------------------------------------------------------------------------ conflict, noise
NOISE_ROWS = [  # label, lo, mid, hi, group, kind (gradconf/scripts/figs.py fig_noise; GRADIENT_CONFLICT_PLAN.md 5.2)
    ("ORCA, same inputs as OMol25 (n = 12)", None, 0.7, None, "dft", "upper"),
    ("second code, same level (n = 4,866), median", None, 1.37, None, "dft", "pt"),
    ("SPICE vs tight recompute (lit.)", None, 1.7, None, "dft", "pt"),
    ("without vs with RIJCOSX (n = 10)", 1.9, None, 10.5, "dft", "rng"),
    ("intermolecular, PBE0+MBD vs ωB97M-V", None, 1.2, None, "dft", "upper"),
    ("all atoms, PBE0+MBD vs ωB97M-V (n = 257)", None, 55.4, None, "dft", "pt"),
    ("wrong SCF state, ~1 % of frames", 20, None, 2000, "dft", "rng"),
    ("1× white level (W)", None, 2.0, None, "cw", "pt"),
    ("T4a / T5 label noise", None, 50.0, None, "cw", "pt"),
    ("A1 on T1, seeds that learn (val RMSE)", 31, None, 36, "cw", "rng"),
    ("A1 on T1, sign-trapped seeds (val RMSE)", 149, None, 214, "cw", "rng"),
]


def build_conflict():
    bf = json.loads((CW_RUN / "gradconf" / "out" / "bf_mined.json").read_text())
    rows = [dict(arm=x["arm"].split("_", 1)[1], system=x["system"], seed=x["seed"], cos_early=x["cos_early"],
                 cos_late=x["cos_late"], Z_err=x["Z"], Z_err_elem=x["Z_elem"], F_err=x["F"],
                 trace=[[int(a), round(b, 4)] for a, b in x["trace"]]) for x in bf]
    variants = {}
    for f in sorted((CW_RUN / "data").glob("T1_x2_*/NOISE.json")):
        name = f.parent.name.replace("T1_x2_", "")
        if "DO_NOT_USE" in name or "check" in name:
            continue
        j = json.loads(f.read_text()); t = j.get("splits", {}).get("train", {})
        variants[name] = dict(ef_m=j["spec"].get("ef_m"), ef_comps=j["spec"].get("ef_comps"), q_m=j["spec"].get("q_m"),
                              q_comps=j["spec"].get("q_comps"), cross=j["spec"].get("cross"),
                              dF_rms=round(t.get("dF_rms_meV_A", np.nan), 2), dE_rms=round(t.get("dE_rms_meV_atom", np.nan), 3),
                              dq_rms=round(t.get("dq_rms_e", np.nan), 4))
    jdump("conflict.json", clean_nan(dict(bf=rows, noise_rows=NOISE_ROWS, variants=variants)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-only", action="store_true")
    a = ap.parse_args()
    if not a.results_only:
        print("tiers"); build_tiers()
        print("structures"); build_structures()
        print("distributions"); build_distributions()
        print("toy"); build_toy()
        print("conflict"); build_conflict()
    print("results"); build_results()
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"data folder {OUT.relative_to(HERE)}: {total / 1e6:.2f} MB")
