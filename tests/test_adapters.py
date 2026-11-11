import json

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
