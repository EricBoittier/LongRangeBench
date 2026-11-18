"""ChargeWorld notebook (notebooks/chargeworld.py): its data folder, the species-charge toy and the 3D widget assets.

The toy must reproduce the stored reference arms exactly (same 64 starts, same trapped mask), the sign flip of every
species must leave the point-charge energy and forces unchanged, and the committed bundle must stay small and free of
machine-specific strings."""
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pytest

NB = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
sys.path.insert(0, str(NB))
import cw_data as C  # noqa: E402

DATA = C.data_dir(NB)
pytestmark = pytest.mark.skipif(not (DATA / "toy.npz").exists(), reason="no ChargeWorld data folder")


@pytest.fixture(scope="module")
def bundle():
    return C.load(DATA)


@pytest.mark.parametrize("arm,kw,combo", [("base", {}, False), ("mb4", dict(batch=4), False), ("combo", {}, True),
                                          ("ema0.999", dict(ema=0.999), False), ("init0.5", dict(init_scale=0.5), False)])
def test_toy_reproduces_reference_arms(bundle, arm, kw, combo):
    D = C.Toy(bundle["toy"], combo=combo)
    c, _ = C.train(D, log_every=0, **kw)
    flipped, _, _ = C.classify(c, D.q0)
    ref = np.asarray(bundle["toy_ref"]["arms"][arm]["trapped"])
    assert np.array_equal(flipped.any(1), ref), (arm, int(flipped.any(1).sum()), int(ref.sum()))


def test_toy_exact_at_reference_charges(bundle):
    D = C.Toy(bundle["toy"])
    L, g = D.full(np.vstack([D.q0, -D.q0]))
    assert L.max() < 1e-6 and np.abs(g).max() < 1e-4


def test_full_batch_matches_per_structure_mean(bundle):
    D = C.Toy(bundle["toy"])
    c = np.random.default_rng(1).normal(0, 0.5, (3, 12))
    L, g = D.full(c); Ls, Gs = D.per_struct(c)
    assert np.allclose(L, Ls.mean(1), rtol=1e-5) and np.allclose(g, Gs.mean(1), rtol=1e-4, atol=1e-6)


def test_global_sign_flip_is_invariant(bundle):
    q0 = np.array([s["q0"] for s in sorted(bundle["tiers"]["species"], key=lambda s: s["Z"])])
    for s in bundle["structures"]["T1"]:
        E0, F0 = C.coulomb_point(s["quad"], q0); E1, F1 = C.coulomb_point(s["quad"], -q0)
        assert abs(E1 - E0) < 1e-9 and np.abs(F1 - F0).max() < 1e-9
        # flipping one charged species present in the structure changes the energy
        charged = [i for i in sorted(set(s["quad"]["sp"])) if abs(q0[i]) > 0.1]
        if not charged:
            continue
        k = charged[0]
        sign = np.ones(12); sign[k] = -1
        assert abs(C.coulomb_point(s["quad"], q0 * sign)[0] - E0) > 1e-3


def test_failure_rule_counts(bundle):
    R = bundle["runs"]["runs"]
    t1 = {x["seed"]: x["fail"] for x in R if x["set"] == "phase1" and x["tier"] == "T1" and x["arm"] == "A1 mid"}
    assert t1 == {13: False, 14: False, 15: True, 16: True}
    assert not any(x["fail"] for x in R if x["set"] == "phase1" and x["tier"] in ("T0", "T2") and x["arm"].startswith("A1"))


def test_cp_interval():
    lo, hi = C.cp_interval(2, 4)
    assert abs(lo - 0.0676) < 1e-3 and abs(hi - 0.9324) < 1e-3
    assert C.cp_interval(0, 4) == (0.0, pytest.approx(0.6024, abs=1e-3))


def test_vendored_three_matches_pin():
    pytest.importorskip("anywidget")
    import cw_widgets as W  # raises on a sha256 mismatch
    assert hashlib.sha256(W.THREE.read_bytes()).hexdigest() == W.THREE_SHA256


#: machine-specific strings that must not leak from the sources into the committed files: absolute paths and
#: scheduler job identifiers (host names are checked by hand before a commit, not listed here)
FORBIDDEN = [r"/home/", r"/scratch", r"/tmp/", r"\bsbatch\b", r"\b39\d{5}\b", r"\.gguf\b"]


def test_bundle_and_sources_are_clean():
    files = list(DATA.iterdir()) + [NB / f for f in ("chargeworld.py", "cw_data.py", "cw_figs.py", "cw_widgets.py",
                                                      "cw_scenes.js")]
    files.append(NB.parent / "tools" / "build_chargeworld_data.py")
    for f in files:
        txt = f.read_bytes().decode("latin-1") if f.suffix != ".npz" else ""
        for pat in FORBIDDEN:
            assert not re.search(pat, txt), (f.name, pat)


def test_bundle_size():
    total = sum(f.stat().st_size for f in DATA.iterdir())
    assert total < 6e6, f"chargeworld data folder is {total / 1e6:.1f} MB"
    json.loads((DATA / "runs.json").read_text())
