import json

import numpy as np

from conftest import LJ, lj
from lrbench.backends import ASEBackend, make_backend
from lrbench.cli import main
from lrbench.data import combine_spins, load_set, parse_term, term_structure
from lrbench.equivariance import check, sample
from lrbench.metrics import Preds, group_value


def test_terms_and_spins():
    assert parse_term("x:y#frags=0,2") == ("x:y", (0, 2))
    assert parse_term("x") == ("x", None)
    assert combine_spins([2, 2]) == 3 and combine_spins([1, 1]) == 1


def test_group_arithmetic(datadir):
    ds = load_set("toy", datadir)
    g = next(g for g in ds.groups if g["name"] == "E_int:AB:o0:d4.0")
    s = term_structure(ds, "AB:o0:d4.0#frag=1")
    assert s.species == ["Ne"] * 3 and s.charge == 0
    P = Preds([{t: {"energy": v} for t, v in [("AB:o0:d4.0", 1.0), ("AB:o0:d4.0#frag=0", 0.25),
                                              ("AB:o0:d4.0#frag=1", 0.5)]}])
    assert np.isclose(group_value(g, P)[0], 0.25)
    assert ds.group_flagged(next(g for g in ds.groups if g["name"] == "E_int:bad"))


def test_eval_lj_is_exact(datadir, tmp_path):
    out = tmp_path / "r.json"
    main(["eval", "--model", LJ, "--sets", "toy", "--data", datadir, "--out", str(out), "--equivariance",
          "--n-boot", "200", "--no-hash"])
    J = json.load(open(out))
    o = J["element_offsets"]["by_level"]["wB97M-V/def2-TZVPD"]
    assert o["residual_mae_meV_atom"] < 1e-6 and all(abs(v) < 1e-6 for v in o["offsets_eV"].values())
    assert J["level"]["model_level"] == "unknown" and not J["level"]["mismatch"]
    R = J["sets"]["toy"]
    E = R["E_int"]
    assert E["n"] == 14 and R["n_pathology_groups"] == 1
    assert all(b["mae"][0] < 1e-6 for b in E["bins"])
    assert abs(E["tail"]["ion_pair"]["slope"][0] - 1) < 1e-6
    w = R["ddE"]["by_family"]["wire_disc_near_vs_far"]
    assert abs(w["slope"][0] - 1) < 1e-6 and w["mae"][0] < 1e-6
    assert R["energy_force"]["all"]["F_mae_meV_A"] < 1e-6
    assert R["pathology"] and all("bad" in (p.get("id") or p.get("name")) for p in R["pathology"])
    eq = json.load(open(out))["equivariance"]["checks"]
    assert set(eq) >= {"rotation", "reflection", "translation", "permutation"}
    assert all(v["max_abs_dE_meV"] < 1e-6 and v["max_force_err_meV_A"] < 1e-6 for v in eq.values())


def test_flat_model_tail_zero(datadir, tmp_path):
    """A model that is LJ truncated at 5 A: no E_int change beyond 8 A -> tail fraction 0."""
    out = tmp_path / "flat.json"
    main(["eval", "--model", "ase:ase.calculators.lj:LennardJones?sigma=2.5&epsilon=0.05&rc=5.0", "--sets", "toy",
          "--data", datadir, "--out", str(out), "--n-boot", "100", "--no-hash", "--label", "flat"])
    t = json.load(open(out))["sets"]["toy"]["E_int"]["tail"]["ion_pair"]["slope"][0]
    assert abs(t) < 1e-6


def test_ensemble_and_compare(datadir, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "ens.json"
    main(["eval", "--model", LJ, "--sets", "toy", "--data", datadir, "--out", str(a), "--n-boot", "100", "--no-hash"])
    main(["eval", "--model", LJ, "--ensemble", LJ.replace("epsilon=0.05", "epsilon=0.051"), "--sets", "toy",
          "--data", datadir, "--out", str(b), "--n-boot", "100", "--no-hash", "--label", "ens"])
    J = json.load(open(b))
    assert J["ensemble"]["n_members"] == 2 and J["ensemble"]["weight_average"]["done"] is False
    assert "calibration" in J["sets"]["toy"]
    main(["compare", str(a), str(b), "--html", str(tmp_path / "c.html"), "--figdir", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "tail_fraction.svg").exists()
    assert (tmp_path / "figs" / "fixedq_wire_parity.png").exists()


def test_equivariance_direct(datadir):
    ds = load_set("toy", datadir)
    smp = sample([term_structure(ds, r["id"]) for r in ds.records], 3)
    r = check(ASEBackend(lj()), smp, n_rot=2)
    assert r["checks"]["rotation"]["max_abs_dE_meV"] < 1e-6


def test_perf_cpu(tmp_path):
    out = tmp_path / "p.json"
    main(["perf", "--model", LJ, "--boxes", "water30", "--warmup", "1", "--n", "2", "--md-steps", "2",
          "--out", str(out)])
    r = json.load(open(out))["rows"][0]
    assert r["n_atoms"] == 30 and r["force_ms"] > 0 and r["md_ms_step"] > 0 and "error" not in r


def test_backend_spec_errors():
    import pytest
    with pytest.raises(ValueError):
        make_backend("model.unknown")


def test_viz(datadir, tmp_path):
    import gzip
    import pytest
    pytest.importorskip("chemiscope")
    from ase.io import write
    from lrbench.data import load_set
    ds = load_set("toy", datadir)
    ext = tmp_path / "ext.extxyz"
    write(str(ext), [term_structure(ds, r["id"]).atoms() for r in ds.records[:4]])
    out = tmp_path / "v.json.gz"
    main(["viz", "--source", f"{datadir}:toy", "--out", str(out), "--model", f"LJ={LJ},{LJ}", "--compare", str(ext)])
    d = json.load(gzip.open(out, "rt"))
    n = len(ds.records) + 4
    assert len(d["structures"]) == n
    P = d["properties"]
    assert set(P["source"]["values"]) == {"toy", "ext.extxyz"}
    assert {"class", "distance", "E_int/REF", "E_int_err/LJ", "E_int_sd/LJ", "tail/REF", "tail/LJ", "locality",
            "flags", "q_formal"} <= set(P)
    assert "none" in P["flags"]["values"] and P["E_int/REF"]["units"] == "meV"
    assert all(len(p["values"]) == (n if p["target"] == "structure" else sum(s["size"] for s in d["structures"]))
               for p in P.values())
    errs = [v for v in P["E_int_err/LJ"]["values"] + P["tail_err/LJ"]["values"] if v == v]
    assert errs and max(abs(v) for v in errs) < 1e-6


def test_level_mismatch_refused(datadir, tmp_path):
    import pytest
    out = tmp_path / "lv.json"
    base = ["eval", "--model", LJ, "--sets", "toy", "--data", datadir, "--out", str(out), "--n-boot", "50",
            "--no-hash", "--model-level", "PBE0+MBD"]
    with pytest.raises(SystemExit, match="level"):
        main(base)
    assert not out.exists()
    main(base + ["--allow-level-mismatch"])
    J = json.load(open(out))
    assert J["level"]["mismatch"] == {"toy": ["wB97M-V/def2-TZVPD"]}
    assert J["notes"][0].startswith("LEVEL MISMATCH")
    from lrbench.report import markdown
    assert "level mismatch" in markdown([J])


def test_viz_levels(datadir, tmp_path):
    import gzip
    import pytest
    pytest.importorskip("chemiscope")
    from ase.io import write
    ds = load_set("toy", datadir)
    ats = []
    for r in ds.records[:3]:
        a = term_structure(ds, r["id"]).atoms()
        a.info.update(level="PBE0+MBD", E_int=-0.1)
        ats.append(a)
    ext = tmp_path / "pbe0.extxyz"
    write(str(ext), ats)
    out = tmp_path / "lv.json.gz"
    main(["viz", "--source", f"{datadir}:toy", "--out", str(out), "--compare", str(ext)])
    P = json.load(gzip.open(out, "rt"))["properties"]
    assert "E_int/REF@wB97M-V/def2-TZVPD" in P and "E_int/REF@PBE0+MBD" in P and "E_int/REF" not in P
    assert set(P["level"]["values"]) == {"wB97M-V/def2-TZVPD", "PBE0+MBD"}


def test_viz_without_chemiscope_matches(datadir, tmp_path, monkeypatch):
    import builtins
    import pytest
    pytest.importorskip("chemiscope")
    from lrbench import viz
    E, _ = viz.entries(f"{datadir}:toy")
    ref = viz.build(E, "x")
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError(n))
                        if n == "chemiscope" else real(n, *a, **k))
    plain = viz.build(E, "x")
    assert plain["structures"] == ref["structures"]
    assert set(plain["properties"]) == set(ref["properties"])
    assert plain["environments"] == ref["environments"]


def test_normalise(datadir, tmp_path):
    out, flat = tmp_path / "n.json", tmp_path / "nf.json"
    args = ["--sets", "toy", "--data", datadir, "--n-boot", "100", "--no-hash", "--normalise", "--distance-unit",
            "bohr", "--energy-unit", "kBT", "--eps-floor", "1e-6"]
    main(["eval", "--model", LJ, "--out", str(out)] + args)
    N = json.load(open(out))["sets"]["toy"]["normalised"]
    t = N["dimer_tail"]
    assert t["phys_source"] == {"E_mp": t["n_points"]} and t["n_points"] == 6
    assert all(abs(r["median"][0]) < 1e-6 for r in t["by_class"].values())
    assert abs(t["marks"]["receptive_field"] - 8 / 0.529177210903) < 1e-3
    assert abs(N["ddE"]["by_series"]["wire_disc_near_vs_far"]["median"][0]) < 1e-6
    assert "skipped" in N["coulomb_prior"]  # LJ has no LES gate
    main(["eval", "--model", "ase:ase.calculators.lj:LennardJones?sigma=2.5&epsilon=0.05&rc=5.0", "--out", str(flat)]
         + args)
    t = json.load(open(flat))["sets"]["toy"]["normalised"]["dimer_tail"]
    assert abs(t["by_class"]["ion_pair"]["median"][0] + 1) < 1e-6  # a flat (local) model misses the whole tail
    main(["compare", str(out), str(flat), "--figdir", str(tmp_path / "f")])
    assert (tmp_path / "f" / "normalised_tail.svg").exists()
    from lrbench.report import markdown
    assert "normalised tail eps" in markdown([json.load(open(flat))])
    from lrbench.normalise import energy_factor
    assert abs(1 / energy_factor("kBT", 300) - 25.852) < 0.01


def test_response_component_is_the_fixed_charge_family_not_a_pooled_slope():
    """A model that is flat on task B1 must score zero response, whatever other ΔΔE families do.

    ``families_fixedq`` holds several ΔΔE families whose reference magnitudes differ by more than an order of
    magnitude. A single slope fitted through the pooled points is dominated by the few largest, so on the real runs
    models whose median predicted ΔΔE was 0.000–0.001 meV over the 90-group fixed-charge family — flat, no response
    at all — collected pooled slopes of 0.43 to 0.85, while two models that genuinely respond were pulled *down*.
    The component therefore reads the named family and nothing else.
    """
    from lrbench.diagnostics import DDE_RESPONSE_FAMILY, score_components

    def result(pooled, per_family):
        return {"sets": {"dimers": {"classes": {"ion_pair": {
                    "tail": [0.9], "eps": [-0.1], "off8": [10.0], "flat8": 100.0}}},
                "families_fixedq": {"ddE": (pooled, None, None),
                                    "ddE_by_family": {k: (v, None, None) for k, v in per_family.items()}}}}

    flat = result(0.85, {DDE_RESPONSE_FAMILY: 0.0, "zwit_ene_rel48": 4.5})
    assert score_components(flat)["response"] == 0.0, "a pooled slope must not buy credit for a flat model"
    assert score_components(flat)["lrs"] == 0.0, "no response at all means no long-range score"

    real = result(0.72, {DDE_RESPONSE_FAMILY: 1.03, "zwit_ene_rel48": 1.75})
    assert score_components(real)["response"] == 1.0, "a responding model must not be dragged down by pooling"
    assert score_components(real)["lrs"] > 0

    # the family missing is "not measured", not "passed": the score says so by being partial
    absent = score_components(result(0.9, {"zwit_ene_rel48": 4.5}))
    assert absent["response"] is None and absent["partial"] is True


def test_reference_only_set_carries_its_level(datadir, tmp_path):
    """A set whose labels live only on the groups (no per-record energy) still has a level, so a cross-level eval
    of it is refused or flagged rather than silently passed."""
    import pytest
    from lrbench.data import load_set
    ds = load_set("toy", datadir)
    ref_only = tmp_path / "ref_only"
    ref_only.mkdir()
    with open(ref_only / "refonly.jsonl", "w") as f:
        for r in ds.records:
            f.write(json.dumps({**r, "set": "refonly", "energy": None, "forces": None, "level": "CCSD(T)/CBS"}) + "\n")
    with open(ref_only / "refonly.groups.json", "w") as f:
        json.dump(ds.groups, f)
    assert load_set("refonly", str(ref_only)).levels() == ["CCSD(T)/CBS"]
    out = tmp_path / "ro.json"
    with pytest.raises(SystemExit, match="level"):
        main(["eval", "--model", LJ, "--sets", "refonly", "--data", str(ref_only), "--out", str(out),
              "--n-boot", "20", "--no-hash", "--model-level", "PBE0+MBD"])
