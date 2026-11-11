"""DFT-emulation rule as executable tests (lrbench.emulation): a charge leak is a deviation from DFT's fragment
charges, not from formal integers; the physics score drops DFT-pathology points, the emulation score keeps them."""
import json

import numpy as np
import pytest

from lrbench import emulation as EM
from lrbench.cli import _fragment_charge_sums
from lrbench.data import load_set

# toy "DFT": an integer-like ion pair at 15 A (Loewdin +-1.00) and a CT-fractional one (Loewdin +-0.80, tagged)
RECS = [
    dict(id="ip:d15", qd=[1.0, -1.0], tags=[], E=-14.399645 / 15.0, d=15.0),
    dict(id="ct:d15", qd=[0.8, -0.8], tags=["ct_fractional"], E=-0.75, d=15.0),
]


@pytest.fixture
def toyset(tmp_path):
    recs, groups = [], []
    for r in RECS:
        x = [[0, 0, 0], [0, 0, r["d"]]]
        recs.append(dict(id=r["id"], set="emulation", species=["Na", "Cl"], positions=x, cell=None, pbc=[False] * 3,
                         charge=0, spin=1, energy=r["E"], forces=None, level="wB97M-V/def2-TZVPD",
                         formal_charges=[1.0, -1.0], fragment_id=[0, 1], fragment_charges=[1, -1], fragment_spins=[1, 1],
                         cls="ion_pair", group=r["id"], role="AB", distance=r["d"],
                         refs=dict(E_int=r["E"], E_A=0.0, E_B=0.0, charges_lowdin=r["qd"]), flags=r["tags"]))
        groups.append(dict(name=f"E_int:{r['id']}", kind="E_int", ref=r["E"],
                           terms=[[r["id"], 1.0], [f"{r['id']}#frag=0", -1.0], [f"{r['id']}#frag=1", -1.0]], meta={}))
    with open(tmp_path / "emulation.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(tmp_path / "emulation.groups.json", "w"))
    return str(tmp_path)


def toy_result(data, charge_rule, energy_rule):
    """A result JSON as `lrbench eval --save-predictions` writes it, from toy per-record charges and energies."""
    ds = load_set("emulation", data)
    preds = {"emulation": {}}
    for r in ds.records:
        q = charge_rule(r)
        preds["emulation"][r["id"]] = {"energy": energy_rule(r), "charges": np.asarray(q, float)}
        for k in (0, 1):
            preds["emulation"][f"{r['id']}#frag={k}"] = {"energy": 0.0, "charges": np.asarray([r["fragment_charges"][k]], float)}
    P = {t: [v["energy"]] for t, v in preds["emulation"].items()}
    return {"label": "toy", "data": {"emulation": {"data": data}},
            "sets": {"emulation": {"predictions": P, "qfrag": _fragment_charge_sums(ds, [preds], "emulation")}}}


dft_q = lambda r: r["refs"]["charges_lowdin"]
formal_q = lambda r: [1.0, -1.0]
leaky_q = lambda r: [v * 0.7 for v in r["refs"]["charges_lowdin"]]  # 30 % of the fragment charge leaks away
dft_E = lambda r: r["energy"]
integer_physics_E = lambda r: -14.399645 / r["distance"]  # integer charges, bare Coulomb: right only where DFT is integer


def kind_pass(scores, kind):
    return scores["toy"]["kind"][kind]["pass_rate_vs_dft"]


def test_dft_charges_pass_everywhere(toyset):
    s = EM.score_rows(EM.rows_from_result(toy_result(toyset, dft_q, dft_E), toyset))
    assert kind_pass(s, "integer-like") == 1.0 and kind_pass(s, "fractional/CT") == 1.0


def test_formal_integers_fail_on_ct_points_only(toyset):
    s = EM.score_rows(EM.rows_from_result(toy_result(toyset, formal_q, integer_physics_E), toyset))
    assert kind_pass(s, "integer-like") == 1.0
    assert kind_pass(s, "fractional/CT") == 0.0
    assert s["toy"]["kind"]["fractional/CT"]["pass_rate_vs_formal"] == 1.0  # formal deviation looks perfect: the trap


def test_leak_beyond_dft_fails(toyset):
    s = EM.score_rows(EM.rows_from_result(toy_result(toyset, leaky_q, dft_E), toyset))
    assert kind_pass(s, "integer-like") == 0.0 and kind_pass(s, "fractional/CT") == 0.0


def test_emulation_vs_physics_energy_scores(toyset):
    em = EM.energy_scores(toy_result(toyset, dft_q, dft_E), toyset)["emulation"]["all"]
    assert em["emulation"]["MAE_meV"] == pytest.approx(0.0) and em["emulation"]["n"] == 2 and em["physics"]["n"] == 1
    ph = EM.energy_scores(toy_result(toyset, formal_q, integer_physics_E), toyset)["emulation"]["all"]
    assert ph["physics"]["MAE_meV"] == pytest.approx(0.0, abs=1e-6)  # integer physics is exact where DFT is integer
    assert ph["emulation"]["MAE_meV"] > 50.0  # but misses DFT's CT point: a good physics score, a poor emulation score


def test_size_consistency_against_dft_decay():
    dft = [dict(sys="X", d=100.0, E_int_meV=-0.5), dict(sys="X", d=32.0, E_int_meV=-5.0)]
    ok = EM.size_consistency([dict(sys="X", d=100.0, E_int_meV=-0.6), dict(sys="X", d=32.0, E_int_meV=-5.5)], dft)
    bad = EM.size_consistency([dict(sys="X", d=100.0, E_int_meV=-45.0)], dft)
    assert all(v["ok"] for v in ok.values()) and not bad["X|100"]["ok"]
