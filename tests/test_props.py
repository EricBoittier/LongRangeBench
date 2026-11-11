"""props set: point-charge moments, scoring against references, and the CLI path (a charge-free model is N/A)."""
import json
import subprocess
import sys

import numpy as np

from lrbench.metrics import Preds
from lrbench.props import BOHR, moments, score
from lrbench.data import DataSet


def _mol(rid, sp, x, q, charge=0):
    mu, th = moments(sp, x, q)
    return dict(id=rid, set="props", species=sp, positions=np.asarray(x, float).tolist(), cell=None, pbc=[False] * 3,
                charge=charge, spin=1, energy=None, level="wB97M-V/def2-TZVPD", role="mol", group=rid, flags=[],
                refs={"dipole_au": mu.tolist(), "quadrupole_au": th.tolist(), "alpha_au": (np.eye(3) * 9.8).tolist(),
                      "charges_lowdin": list(q)}, extra={"source": "test"})


def _scan(d):
    x = [[0, 0, 0], [d, 0, 0], [d + 0.6, 0.76, 0], [d + 0.6, -0.76, 0]]
    q = [1.0, -0.8 - 0.1 / d, 0.4 + 0.05 / d, 0.4 + 0.05 / d]
    return dict(id=f"scan:{d}", set="props", species=["Na", "O", "H", "H"], positions=x, cell=None, pbc=[False] * 3,
                charge=1, spin=1, energy=None, level="wB97M-V/def2-TZVPD", role="scan", group="Na_H2O:o0", cls="ion_polar",
                distance=d, fragment_id=[0, 1, 1, 1], fragment_charges=[1, 0], refs={"charges_lowdin": q}, flags=[])


def _ds():
    recs = [_mol("w", ["O", "H", "H"], [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]], [-0.8, 0.4, 0.4]),
            _mol("ion", ["N", "H", "H", "H", "H"], [[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [0.6, -0.6, -0.6],
                                                   [-0.6, 0.6, -0.6]], [-0.6, 0.4, 0.4, 0.4, 0.4], charge=1)]
    recs += [_scan(d) for d in (3.0, 5.0, 8.0, 12.0, 25.0)]
    return DataSet("props", recs, [], {})


def test_moments_point_dipole():
    mu, th = moments(["H", "H"], [[0, 0, 0], [1.0, 0, 0]], [1.0, -1.0])
    assert np.allclose(mu, [-1.0 / BOHR, 0, 0])
    assert abs(np.trace(th)) < 1e-12


def test_exact_charges_score_zero():
    ds = _ds()
    member = {r["id"]: {"energy": None, "charges": np.array(r["refs"]["charges_lowdin"])} for r in ds.records}
    S = score(ds, Preds([member]))
    assert S["dipole"]["all"]["vector_err"]["MAE"] < 1e-12 and abs(S["dipole"]["all"]["slope"] - 1) < 1e-12
    assert S["quadrupole"]["all"]["norm_err"]["MAE"] < 1e-12
    assert S["polarisability"]["status"].startswith("N/A")
    ind = S["induced_dipole"]
    assert ind["vector_err_by_region"]["<8"]["MAE"] < 1e-12 and ind["curves"][0]["dmu_ref_au"][-1] == 0


def test_half_charges_slope():
    ds = _ds()
    member = {r["id"]: {"energy": None, "charges": 0.5 * np.array(r["refs"]["charges_lowdin"])} for r in ds.records}
    assert abs(score(ds, Preds([member]))["dipole"]["neutral"]["slope"] - 0.5) < 1e-12


def test_cli_props_na(tmp_path):
    ds = _ds()
    d = tmp_path / "canonical"; d.mkdir()
    with open(d / "props.jsonl", "w") as f:
        for r in ds.records:
            f.write(json.dumps(r) + "\n")
    (d / "props.groups.json").write_text("[]")
    out = tmp_path / "p.json"
    subprocess.run([sys.executable, "-m", "lrbench", "eval", "--model", "ase:ase.calculators.lj:LennardJones?sigma=2.5",
                    "--model-level", "wB97M-V/def2-TZVPD", "--sets", "props", "--data", str(d), "--no-hash",
                    "--out", str(out)], check=True, capture_output=True)
    R = json.load(open(out))
    assert R["sets"]["props"]["status"].startswith("N/A")
