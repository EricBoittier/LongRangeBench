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
