import json

import pytest

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from conftest import lj
from lrbench.adapters import Adapter, default_level
from lrbench.backends import ASEBackend
from lrbench.cli import main
from lrbench.data import Structure, default_model_level


class Echo(Calculator):
    """LJ energy plus what the wrapped model would see: Q and 2S+1 under the inner model's own keys."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy()
        a.calc = lj()
        n = len(atoms)
        self.seen = dict(atoms.info)
        self.results = {"energy": a.get_potential_energy(), "forces": a.get_forces(),
                        "partial_charges": np.full(n, atoms.info.get("charge", 0) / n)}


def _s(q=1, spin=2):
    return Structure("x", ["Ne", "Ne", "Ne"], np.array([[0, 0, 0], [3.0, 0, 0], [0, 3.0, 0]]), q, spin)


def test_levels():
    assert default_level("ase:lrbench.adapters:so3lr?model='so3lr-tauri-m'") == "PBE0+MBD-NL"
    assert default_level("ase:lrbench.adapters:mace_mh?head='omat_pbe'").startswith("PBE")
    assert default_model_level("ase:lrbench.adapters:mace_polar") == "wB97M-V/def2-TZVPD"
    assert default_model_level("ase:ase.calculators.lj:LennardJones") == "unknown"


def test_adapter_maps_charge_spin_and_returns_charges_dipole():
    inner = Echo()
    ad = Adapter(inner, "L", {}, info_map={"spin_multiplicity": "multiplicity"}, extra_info={"external_field": [0] * 3},
                 charges=lambda R: R.get("partial_charges"))
    p = ASEBackend(ad).predict([_s()])[0]
    assert p["error"] is None and inner.seen["multiplicity"] == 2 and inner.seen["charge"] == 1
    assert inner.seen["external_field"] == [0] * 3
    assert np.isclose(p["charges"].sum(), 1.0)
    assert np.allclose(p["dipole"], (p["charges"][:, None] * _s().positions).sum(0))
    ref = _s().atoms()
    ref.calc = lj()
    assert np.isclose(p["energy"], ref.get_potential_energy())


def test_adapter_without_charge_spin_hides_them():
    inner = Echo()
    ASEBackend(Adapter(inner, "L", {}, charge_spin=False)).predict([_s()])
    assert "charge" not in inner.seen and "spin" not in inner.seen


def test_predict_writes_eval_json(tmp_path):
    from ase.io import write
    a = _s(q=0, spin=1).atoms()
    write(tmp_path / "in.xyz", [a, a], format="extxyz")
    out = tmp_path / "o.json"
    main(["predict", "--model", "ase:ase.calculators.lj:LennardJones?sigma=2.5&epsilon=0.05&rc=40.0",
          str(tmp_path / "in.xyz"), str(out)])
    J = json.load(open(out))
    assert J["n_frames"] == 2 and J["n_errors"] == 0 and len(J["frames"][0]["forces"]) == 3


# ---------------------------------------------------------------- charge/spin must not be served from a stale cache
class _ChargeSensitive(Calculator):
    """A calculator whose energy depends only on the total charge, so a cached result is unmistakable."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self):
        super().__init__()
        self.n_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.n_calls += 1
        q = float(self.atoms.info.get("charge", 0))
        self.results = {"energy": 7.0 * q, "free_energy": 7.0 * q,
                        "forces": np.zeros((len(self.atoms), 3))}


def _struct(key, charge, spin=1):
    return Structure(key=key, species=["Ar", "Ar"], positions=np.array([[0.0, 0, 0], [3.0, 0, 0]]),
                     charge=charge, spin=spin)


def test_charge_states_of_one_geometry_are_not_cached():
    """ASE's check_state ignores atoms.info, so without an explicit cache drop the second structure here would be
    served the first one's energy. charge_parity is made entirely of such pairs."""
    calc = _ChargeSensitive()
    be = ASEBackend(calc, label="charge-sensitive")
    res = be.predict([_struct("q0", 0), _struct("qm1", -1), _struct("qp1", +1), _struct("q0b", 0)])
    assert [r["energy"] for r in res] == [0.0, -7.0, 7.0, 0.0]
    assert calc.n_calls == 4


def test_adapter_recomputes_when_only_the_charge_changes():
    """The same trap one level down: the Adapter must not serve its own cache either, and must invalidate the
    calculator it wraps."""
    from ase import Atoms

    inner = _ChargeSensitive()
    ad = Adapter(inner, "level", {"model": "stub"})
    seen = []
    for q in (0, -1, 0, 2):
        a = Atoms("Ar2", positions=[[0, 0, 0], [3, 0, 0]])
        a.info.update(charge=q, spin_multiplicity=1)
        a.calc = ad
        seen.append(a.get_potential_energy())
    assert seen == [0.0, -7.0, 0.0, 14.0]


def test_metatomic_level_reads_the_dataset_from_the_name():
    """The upet names carry their training set, and the longest token must win: ``pet-omatpes-l`` is OMatPES, not
    OMat24, and reading it as the latter would silently score it against the wrong labels."""
    from lrbench.adapters import OMOL, SPICE, metatomic_level
    assert metatomic_level("checkpoints/pet-omol-s.pt") == OMOL
    assert metatomic_level("pet-omol-l-v1.0.0.pt") == OMOL
    assert metatomic_level("pet-spice-s.pt") == SPICE
    assert metatomic_level("pet-mad-s.pt") == "PBEsol/PAW (MAD)"
    assert metatomic_level("pet-omat-l.pt") == "PBE/PAW (OMat24)"
    assert metatomic_level("pet-omatpes-l.pt") == "mixed"   # not read as 'omat'
    assert metatomic_level("some-other-model.pt") == "unknown"


class _FakeAtomisticModel:
    def __init__(self, inputs):
        self._inputs = dict.fromkeys(inputs, object())

    def requested_inputs(self, use_new_names=True):
        return self._inputs


class _FakeMetatomicCalc(Calculator):
    """Stands in for MetatomicCalculator: an LJ energy plus the ``_model`` whose declared inputs the adapter reads."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, inputs):
        super().__init__()
        self._model = _FakeAtomisticModel(inputs)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy()
        a.calc = lj()
        self.seen = dict(atoms.info)
        self.results = {"energy": a.get_potential_energy(), "forces": a.get_forces()}


def test_metatomic_charge_spin_follows_what_the_model_declares(monkeypatch):
    """``charge_spin`` is read off ``requested_inputs``, never guessed.

    A model that asks for a charge is given one and scored on what it does with it; a model that asks for nothing is
    recorded as unable to be *told* one, so "ignored the charge" keeps meaning only what it says. Of the released
    PET checkpoints only the pet-omol-* family requests ``charge``.
    """
    import lrbench.backends as B
    from lrbench.adapters import metatomic

    for inputs, conditioned in ((["charge", "spin_multiplicity"], True), ([], False)):
        monkeypatch.setattr(B, "metatomic_calculator", lambda path, device, _i=inputs: _FakeMetatomicCalc(_i))
        ad = metatomic("pet-omol-s.pt", device="cpu")
        assert ad.charge_spin is conditioned
        assert ad.lrbench_meta["requested_inputs"] == list(inputs)
        assert ("NONE" in ad.lrbench_meta["charge_spin"]) is not conditioned
        p = ASEBackend(ad).predict([_s()])[0]
        assert p["error"] is None
        # a model that declares no charge input is never handed one
        assert ("charge" in ad.inner.seen) is conditioned


def test_metatomic_fails_safe_when_the_model_cannot_be_asked(monkeypatch):
    """``_model`` is private, so not finding it must mean "cannot ask", never "takes no charge".

    The two mistakes are not symmetric. Stripping the charge from a model that needs one turns an ion pair neutral
    and produces a confidently flat tail -- a wrong number that looks like a finding. Leaving the charge in for a
    model that does not want it costs nothing, because metatomic forwards only the inputs the model asks for. So the
    charge stays, the record says ``UNKNOWN`` rather than ``NONE``, and the user is warned.
    """
    import lrbench.backends as B
    from lrbench.adapters import metatomic

    class Mute(_FakeMetatomicCalc):
        def __init__(self):
            super().__init__([])
            self._model = None

    monkeypatch.setattr(B, "metatomic_calculator", lambda path, device: Mute())
    with pytest.warns(UserWarning, match="requested_inputs"):
        ad = metatomic("mystery.pt", device="cpu", level="L")
    assert ad.charge_spin is True, "an unreadable model must keep its charge, not lose it"
    assert ad.lrbench_level == "L"
    assert ad.lrbench_meta["requested_inputs"] is None          # unknown, not "empty"
    assert ad.lrbench_meta["charge_spin"].startswith("UNKNOWN")
    assert ASEBackend(ad).predict([_s()])[0]["error"] is None


def test_metatomic_records_the_package_version(monkeypatch):
    """The distribution is ``metatomic-torch``; asking for ``metatomic`` records the version as '?'."""
    import lrbench.backends as B
    from lrbench.adapters import metatomic

    monkeypatch.setattr(B, "metatomic_calculator", lambda path, device: _FakeMetatomicCalc(["charge"]))
    meta = metatomic("pet-omol-s.pt", device="cpu").lrbench_meta
    assert meta["package"] == "metatomic-torch" and meta["version"] != "?"
