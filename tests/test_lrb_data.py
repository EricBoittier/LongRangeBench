"""Species-charge parsing of the notebooks' data layer (notebooks/lrb_data.py): the multivalent switch keys on it.

Every species name that appears in the committed bundle (notebooks/data/lr_program/) must be listed in EXPECTED, so
a new ion added to the bundle fails here until its charge is written down."""
import json
import pathlib
import sys

import numpy as np
import pytest

NB = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
sys.path.insert(0, str(NB))
import lrb_data as L  # noqa: E402

BUNDLE = NB / "data" / "lr_program"

#: |charge| of every species name used in pair keys, plus the forms the parser must also read
EXPECTED = {
    "NH4+": 1, "Gdm+": 1, "K+": 1, "Na+": 1, "Li+": 1, "Mg2+": 2, "Ca2+": 2, "SO4 2-": 2, "NO3-": 1, "OAc-": 1,
    "F-": 1, "Cl-": 1, "I-": 1, "Br-": 1, "H2O": 0, "MeCN": 0, "SO4(2-)": 2, "(SO4)2-": 2, "[Fe(H2O)6]3+": 3,
    "Mg++": 2, "SO4--": 2, "Zn2+": 2, "PF6-": 1, "BF4-": 1, "TMA+": 1,
}
#: bare ion names of the ladder / MD blocks (no sign; read through lrb_data.ION_Q)
EXPECTED_BARE = {"Na": 1, "Cl": 1, "Br": 1, "K": 1, "Li": 1, "Mg": 2, "Ca": 2, "Zn": 2, "W": 0, "False": 0}


@pytest.mark.parametrize("name,q", sorted(EXPECTED.items()))
def test_species_charge(name, q):
    assert L.species_charge(name) == q


def test_pair_maxq_examples():
    assert L.pair_maxq("NH4+_NH4+") == 1
    assert L.pair_maxq("Gdm+_Mg2+") == 2
    assert L.pair_maxq("H2O_H2O") == 0
    assert L.pair_maxq("K+_MeCN") == 1


def _bundle_pair_species():
    names = set()
    for blk in ("typeheld", "setB", "search"):
        p = BUNDLE / f"chem_{blk}.npz"
        if p.exists():
            names |= {sp for pair in np.load(p)["pair"] for sp in str(pair).split("_")}
    p = BUNDLE / "typeheld_agg.json"
    if p.exists():
        for v in json.loads(p.read_text()).values():
            for tag in ("on", "off"):
                names |= {sp for k in (v.get(tag) or {}) if "_" in k and " " not in k for sp in k.split("_")}
    p = BUNDLE / "program.json"
    if p.exists():
        S = json.loads(p.read_text()).get("scans") or {}
        names |= {sp for k in (S.get("search") or {}) for sp in k.split("|")[0].split("_")}
        names |= {sp for k in (S.get("mae") or {}) for sp in k.split("|")[1].split("_") if "(" not in sp}
    return names


def _bundle_bare_ions():
    names = set()
    for blk, key in (("ladder", "ion"), ("md", "sys")):
        p = BUNDLE / f"chem_{blk}.npz"
        if p.exists():
            names |= set(map(str, np.load(p)[key]))
    return names


@pytest.mark.skipif(not BUNDLE.exists(), reason="no committed bundle")
def test_every_bundle_species_is_listed_and_parsed():
    names = _bundle_pair_species()
    assert names, "no species names found in the bundle"
    missing = sorted(n for n in names if n not in EXPECTED)
    assert not missing, f"species in the bundle without an expected charge: {missing}"
    for n in names:
        assert L.species_charge(n) == EXPECTED[n], n


@pytest.mark.skipif(not BUNDLE.exists(), reason="no committed bundle")
def test_every_bundle_bare_ion_is_listed():
    names = _bundle_bare_ions()
    missing = sorted(n for n in names if n not in EXPECTED_BARE)
    assert not missing, f"bare ion names in the bundle without an expected charge: {missing}"
    for n in names:
        assert L.ION_Q.get(n, 1) == EXPECTED_BARE[n], n


# ----------------------------------------------------------------------------------- large-n ladder: absolute rule
TOOLS = NB.parent / "tools"


def _builder():
    sys.path.insert(0, str(TOOLS))
    import build_lr_program_data as B
    return B


def _fake_rows(offset, ion="Na"):
    """one seed, error = a constant offset at every n (a flat curve: zero slope for any offset)"""
    return [dict(err_mean={n: offset for n in (200, 500, 1000, 2000, 4000)})]


def test_constant_offset_passes_the_slope_but_not_the_absolute_rule():
    B = _builder()
    for ion in ("Na", "Cl"):
        assert B.a1_summary(_fake_rows(0.3), ion)["abs_pass_all"], "a 0.3 eV offset is inside the draft limit"
        for c in (3.0, 14.0, 30.0, -30.0):
            assert not B.a1_summary(_fake_rows(c), ion)["abs_pass_all"], (ion, c)
            assert B._ladder_slope({n: [c] for n in (1000, 2000, 4000)}) == pytest.approx(0.0, abs=1e-12)


def test_draft_limits_are_twice_the_fit_uncertainty_values_of_the_spec():
    B = _builder()
    assert B.A1_LIMIT_EV == {"Na": {1000: 0.86, 4000: 1.14}, "Cl": {1000: 1.09, 4000: 1.36}}


@pytest.mark.skipif(not (BUNDLE / "program.json").exists(), reason="no committed bundle")
def test_committed_large_n_ladder_uses_the_absolute_rule():
    P = json.loads((BUNDLE / "program.json").read_text())
    LB = P["ladder_large_n"]
    assert LB["limit_status"] == "draft" and "limit" not in LB
    for arm, ions in LB["arms"].items():
        for ion, x in ions.items():
            s = x["summary"]
            assert s["abs_pass_seeds"] == 0, (arm, ion)       # no arm is inside the limit (AL, PAL, P, C0 all 10-200 eV off)
    txt = " ".join(LB["text"]) + " " + LB["note"] + " " + " ".join(P["evidence_limits"])
    assert " passes the slope rule (rule 3)" not in txt and "slope rule is passed by" not in txt
    assert "Seeds inside the limit at both n" in txt
