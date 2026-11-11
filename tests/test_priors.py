"""Priors suite: every check catches its planted defect in a synthetic analytic model and passes the clean point-charge
model; zero scorable points is never a pass."""
import json

import numpy as np
import pytest

from lrbench.backends import ASEBackend
from lrbench.data import Structure, load_set
from lrbench.priors import FAIL, NA, NE, ORDER, PASS, WARN, Ctx, run
from lrbench.priors import synthetic as S

LEVEL = "wB97M-V/def2-TZVPD"

# planted defect -> the checks it must fail; ALLOWED: further failures that follow from the same defect
INTENDED = {
    "Switched": {"seam", "nve_short"},
    "Leaky": {"fragment_charges", "size_consistency"},
    "Caching": {"charge_conditioning"},
    "Flat": {"far_field_exponent", "eps_far", "offset_free_degeneracy"},
    "SpuriousLR": {"locality"},
    "SepOffset": {"size_consistency", "charge_offset_by_Q"},
    "QBlind": {"charge_conditioning", "charge_state"},
    "IPShift": {"charge_state"},
    "NonConservative": {"energy_force", "conservativeness"},
    "Discontinuous": {"energy_force", "seam"},
    "NonInvariant": {"invariance"},
    "Noisy": {"determinism"},
    "BoxDependent": {"periodic_open"},
}
ALLOWED = {
    "Flat": {"size_consistency"},  # no Coulomb at 100 A
    "SpuriousLR": {"far_field_exponent", "eps_far", "offset_free_degeneracy"},  # the 1/r term cancels Na+-H2O's span
    "NonConservative": {"nve_short"},
    "NonInvariant": {"periodic_open"},  # boxing translates the structure
    # Noise breaks the gradient relation as surely as it breaks everything else it touches.
    "Noisy": {"far_field_exponent", "energy_force", "invariance", "locality", "charge_conditioning",
              "precision", "conservativeness", "virial_tail"},
    "QBlind": {"charge_offset_by_Q"},
    "Caching": {"charge_state"},
    "Discontinuous": {"nve_short", "size_consistency"},  # the jump persists to 100 A and kicks the NVE pair
    # The offset switches on when the clusters split at 3 A: itself a jump. virial_tail sees it too,
    # because a constant offset on the separated fragments stops E_int decaying to zero, so r F / E_int
    # is measured against the wrong denominator. That is a real sensitivity, not a false alarm.
    "SepOffset": {"energy_force", "virial_tail"},
    "Leaky": {"far_field_exponent", "eps_far", "charge_offset_by_Q"},  # cached states give IP = EA = 0 (the GSCDB IP/EA pairs the upet bug hit)
}


@pytest.fixture(scope="module")
def dimers(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("syn"))
    S.write_dimer_set(d)
    return load_set("dimers", d)


def _ev(calc, sp, x, q, s=1):
    from ase import Atoms
    a = Atoms(sp, positions=x)
    a.info.update(charge=q, spin_multiplicity=s)
    a.calc = calc
    return float(a.get_potential_energy())


W = (["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])


@pytest.fixture(scope="module")
def ipea():
    c = S.PointCharge()
    mols = [(["Na"], [[0, 0, 0]], 1), (["Cl"], [[0, 0, 0]], -1), (W[0], W[1], 0),
            (["Na"] + W[0], [[0, 0, -2.4]] + W[1], 1)]
    out = []
    for k, (sp, x, q) in enumerate(mols):
        x = np.asarray(x, float)
        e0, em, ep = _ev(c, sp, x, q), _ev(c, sp, x, q + 1, 2), _ev(c, sp, x, q - 1, 2)
        out.append({"frag": f"m{k}", "structure": Structure(f"ipea:m{k}", sp, x, q, 1), "IP_eV": em - e0,
                    "EA_eV": e0 - ep, "mult_states": {"Nm1": 2, "Np1": 2}, "flags": []})
    return out


@pytest.fixture(scope="module")
def sepframes():
    c = S.PointCharge()
    out = []
    for d in (13.0, 15.0, 17.0, 20.0):  # Na+ ... H2O, one charged component far from a neutral one
        sp, x = ["Na"] + W[0], np.vstack([[[0, 0, -d]], np.asarray(W[1])])
        out.append({"structure": Structure(f"sep{d}", sp, x, 1, 1), "E_dft": _ev(c, sp, x, 1), "ref_offset": 0.0,
                    "stratum": "separated"})
    for d in (2.3, 2.5, 2.7):  # the same species in contact: one component
        sp, x = ["Na"] + W[0], np.vstack([[[0, 0, -d]], np.asarray(W[1])])
        out.append({"structure": Structure(f"one{d}", sp, x, 1, 1), "E_dft": _ev(c, sp, x, 1), "ref_offset": 0.0,
                    "stratum": "single"})
    return out


def _ctx(model, dimers, **kw):
    kw.setdefault("ref_backend", ASEBackend(model, label="self"))  # precision: the model against itself
    return Ctx(backend=ASEBackend(model, label=type(model).__name__), dimers=dimers, model_level=LEVEL,
               noise_rel=1e-12, cutoff=8.0, **kw)


def _matrix(name, dimers, ipea, sepframes):
    m = S.ALL[name]()
    ref = S.ALL[name]() if name != "Noisy" else S.Noisy(seed=1)
    return run(_ctx(m, dimers, ipea=ipea, sepframes=sepframes, ref_backend=ASEBackend(ref)), ORDER, log=lambda s: None)


def test_clean_passes_everything(dimers, ipea, sepframes):
    r = run(_ctx(S.PointCharge(), dimers, ipea=ipea, sepframes=sepframes), ORDER, log=lambda s: None)
    st = {k: v["status"] for k, v in r.items()}
    assert not [k for k, v in st.items() if v == FAIL], st
    must_pass = set(ORDER) - {"contamination", "distribution_shift", "ct_unbound_labelled"}
    assert all(st[k] == PASS for k in must_pass), st


@pytest.mark.parametrize("name", sorted(INTENDED))
def test_planted_defect_caught_and_only_it(name, dimers, ipea, sepframes):
    r = _matrix(name, dimers, ipea, sepframes)
    fails = {k for k, v in r.items() if v["status"] == FAIL}
    assert INTENDED[name] <= fails, (name, fails)
    assert fails <= INTENDED[name] | ALLOWED.get(name, set()), (name, fails - INTENDED[name])


def test_steep_conservative_ramp_is_not_an_energy_force_failure(dimers):
    r = run(_ctx(S.Switched(ramp=0.004), dimers), ["energy_force"], log=lambda s: None)["energy_force"]
    assert r["status"] == PASS and r["n_unresolved"] > 0, r["summary"]


def test_seam_reports_the_window(dimers):
    r = run(_ctx(S.Switched(), dimers), ["seam"], log=lambda s: None)["seam"]
    lo, hi = r["windows_by_class"]["ion_pair"]
    assert 7.2 <= lo <= 7.6 and 8.9 <= hi <= 9.3


def test_size_consistency_names_total_charge_rows(dimers):
    class TotalRows(S.Flat):  # fragments conditioned on the total charge once they are apart
        def charge_state(self, sp, x, Q, Sp, lab, ncl):
            if ncl < 2:
                return super().charge_state(sp, x, Q, Sp, lab, ncl)
            return sum(super(TotalRows, self).charge_state([s for s, l in zip(sp, lab) if l == k], x, Q, Sp, lab, 1)
                       for k in range(ncl))
    r = run(_ctx(TotalRows(), dimers), ["size_consistency"], log=lambda s: None)["size_consistency"]
    assert r["status"] == FAIL
    row = next(p for p in r["pairs"] if p["scan"] == "Li_Na:o0")
    assert row["closest_assignment"] == "(2,2)"


def test_ip_damage_against_baseline(dimers, ipea):
    base = {"label": "clean", "checks": run(_ctx(S.PointCharge(), dimers, ipea=ipea), ["charge_state"],
                                            log=lambda s: None)}
    r = run(_ctx(S.IPShift(shift=0.2), dimers, ipea=ipea, baseline=base), ["charge_state"],
            log=lambda s: None)["charge_state"]
    assert r["status"] == FAIL and "IP damage" in r["summary"]


def test_zero_points_never_pass(ipea):
    r = run(Ctx(backend=ASEBackend(S.PointCharge()), dimers=None, noise_rel=1e-12), ORDER, log=lambda s: None)
    assert not [k for k, v in r.items() if v["status"] == PASS], {k: v["status"] for k, v in r.items()}


def test_far_field_tail_too_short_is_not_evaluable(tmp_path):
    S.write_dimer_set(str(tmp_path), dists=[3.5, 4.0, 5.0, 6.0, 8.0])
    ds = load_set("dimers", str(tmp_path))
    r = run(_ctx(S.PointCharge(), ds), ["far_field_exponent", "charge_offset_by_Q"], log=lambda s: None)
    assert r["far_field_exponent"]["status"] == NE and r["charge_offset_by_Q"]["status"] == NE


def test_no_charges_no_conditioning_not_applicable(dimers):
    class NoQ(S.PointCharge):
        implemented_properties = ["energy", "forces"]

        def calculate(self, *a, **k):
            super().calculate(*a, **k)
            self.results.pop("charges")
    be = ASEBackend(NoQ())
    be.passes_charge_spin = False
    r = run(Ctx(backend=be, dimers=dimers, noise_rel=1e-12), ["fragment_charges", "charge_conditioning",
                                                              "charge_state", "precision"], log=lambda s: None)
    assert all(v["status"] == NA for v in r.values()), {k: v["status"] for k, v in r.items()}


def test_data_checks(dimers):
    far_id = next(r["id"] for r in dimers.records if r["distance"] >= 10)
    cov = [{"g": 8.5, "qa": 1.0, "qb": -1.0}] * 3 + [{"g": 15.0, "cls": "ion_pair"}] * 20
    r = run(_ctx(S.PointCharge(), dimers, seen={far_id: ["train_aug"]}, train_coverage=cov),
            ["contamination", "distribution_shift", "level_of_theory"], log=lambda s: None)
    assert r["contamination"]["status"] == WARN and r["contamination"]["n_seen"] == 1
    assert r["distribution_shift"]["status"] == WARN and "ion_pair 8-9" in r["distribution_shift"]["summary"]
    c = _ctx(S.PointCharge(), dimers)
    c.model_level = "PBE0+MBD"
    assert run(c, ["level_of_theory"], log=lambda s: None)["level_of_theory"]["status"] == FAIL


def test_ct_ambiguous_kept_and_dropped(tmp_path):
    S.write_dimer_set(str(tmp_path), pairs={"Na_Cl": S.PAIRS["Na_Cl"][:-1] + ("ion_pair_ct_ambiguous",),
                                            "Na_H2O": S.PAIRS["Na_H2O"]})
    ds = load_set("dimers", str(tmp_path))
    r = run(_ctx(S.PointCharge(), ds), ["ct_unbound_labelled"], log=lambda s: None)["ct_unbound_labelled"]
    assert r["status"] == PASS and r["classes"]["ct_ambiguous"]["n"] == 10

    class NoCl(S.PointCharge):
        def calculate(self, atoms=None, *a, **k):
            if "Cl" in atoms.get_chemical_symbols():
                raise RuntimeError("element not in the species map")
            super().calculate(atoms, *a, **k)
    r = run(_ctx(NoCl(), ds), ["ct_unbound_labelled"], log=lambda s: None)["ct_unbound_labelled"]
    assert r["status"] == FAIL


def test_precision_against_a_different_reference(dimers):
    r = run(_ctx(S.PointCharge(), dimers, ref_backend=ASEBackend(S.Flat())), ["precision"], log=lambda s: None)
    assert r["precision"]["status"] == FAIL


def test_cli_and_probe_files(tmp_path, ipea, sepframes):
    from ase import Atoms
    from ase.io import write
    from lrbench.cli import main
    S.write_dimer_set(str(tmp_path))
    A = []
    for p in ipea:
        s = p["structure"]
        a = Atoms(s.species, positions=s.positions)
        a.info.update(charge=s.charge, spin_multiplicity=s.spin, frag=p["frag"], IP_eV=p["IP_eV"], EA_eV=p["EA_eV"],
                      mult_Nm1=2, mult_Np1=2)
        A.append(a)
    write(str(tmp_path / "ipea.extxyz"), A)
    F = []
    for f in sepframes:
        s = f["structure"]
        a = Atoms(s.species, positions=s.positions)
        a.info.update(charge=s.charge, spin_multiplicity=s.spin, E_dft=f["E_dft"], ref_offset=0.0, stratum=f["stratum"])
        F.append(a)
    write(str(tmp_path / "sep.extxyz"), F)
    out = tmp_path / "p.json"
    main(["priors", "--model", "ase:lrbench.priors.synthetic:SepOffset", "--data", str(tmp_path), "--out", str(out),
          "--checks", "charge_state,charge_offset_by_Q,size_consistency", "--ipea", str(tmp_path / "ipea.extxyz"),
          "--sep-frames", str(tmp_path / "sep.extxyz"), "--model-level", LEVEL, "--noise-rel", "1e-12",
          "--md", str(tmp_path / "p.md")])
    R = json.load(open(out))
    assert R["checks"]["charge_state"]["status"] == PASS
    assert R["checks"]["charge_offset_by_Q"]["status"] == FAIL
    assert "**FAIL**" in open(tmp_path / "p.md").read()


def test_virial_recovers_the_coulomb_exponent_and_abstains_without_a_tail(dimers):
    """r F_r / E_int = p, so a Coulomb tail reads 1 — measured from the model alone, with no reference.

    The two outcomes that matter are a number and a refusal. An exact point-charge model must recover p = 1 on the
    classes where both fragments are charged, which is the only family whose leading term is monopole-monopole. A
    flat model must come back *not evaluable*: beyond the cutoff it returns neither a force nor an energy to form
    the ratio from, and calling that a pass would turn an absence of physics into a clean bill of health.
    """
    r = run(_ctx(S.PointCharge(), dimers), ["virial_tail"], log=lambda s: None)["virial_tail"]
    assert r["status"] == PASS, r["summary"]
    scored = {c: v for c, v in r["by_class"].items() if v["scored"] and v["n_points"] >= 4}
    assert scored, r["by_class"]
    for cls, v in scored.items():
        assert abs(v["median_p"] - 1.0) < 0.2, (cls, v)
        assert v["expected_p"] == 1
    # an ion against a neutral molecule is charge-dipole or charge-induced-dipole, so p = 1 is not expected there
    # and those classes are reported without being scored
    assert any(not v["scored"] for v in r["by_class"].values())

    flat = run(_ctx(S.Flat(), dimers), ["virial_tail"], log=lambda s: None)["virial_tail"]
    assert flat["status"] == NE, flat
    assert "no tail" in flat["summary"] or "flat tail" in flat["summary"]


def test_conservativeness_separates_a_gradient_from_a_scaled_force(dimers):
    """The check exists to tell a conservative model from one whose forces are a separate head.

    `NonConservative` returns 1.2x the true gradient, so |F + dE/dr| / |F| sits at 1/6 -- well above the
    threshold -- while an exact model sits at numerical noise.
    """
    good = run(_ctx(S.PointCharge(), dimers), ["conservativeness"], log=lambda s: None)["conservativeness"]
    assert good["status"] == PASS, good["summary"]
    for region, v in good["by_region"].items():
        if v["n_points"]:
            assert v["median_rel_deviation"] < 0.02, (region, v)

    bad = run(_ctx(S.NonConservative(), dimers), ["conservativeness"], log=lambda s: None)["conservativeness"]
    assert bad["status"] == FAIL, bad
    worst = max(v["median_rel_deviation"] for v in bad["by_region"].values() if v["n_points"])
    assert worst > 0.1, bad["by_region"]
