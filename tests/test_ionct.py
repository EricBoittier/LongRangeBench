"""The multivalent-ion and charge-transfer panels: per-point tables (lrbench.ionct), the private-data guard, and a
smoke run of the notebook cells that draw them (skipped where marimo is not installed, as in CI)."""
import importlib.util
import json
import os
import pathlib
import sys

import pytest

from lrbench import ionct
from lrbench.diagnostics import collect

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "long_range_diagnostics.py"
LEVEL = "wB97M-V/def2-TZVPD"


def _rec(rid, species, pos, fid, fq, cls, d, flags=(), lowdin=None, spin=1, energy=-1.0, **refs):
    r = {"id": rid, "species": species, "positions": pos, "charge": int(sum(fq)), "spin": spin, "energy": energy,
         "level": LEVEL, "fragment_id": fid, "fragment_charges": fq, "cls": cls, "distance": d, "role": "AB",
         "group": rid.rsplit(":", 1)[0], "flags": list(flags), "refs": dict(refs), "extra": {"pair": rid.split(":")[0]}}
    if lowdin is not None:
        r["refs"]["charges_lowdin"] = lowdin
    return r


def _scan(name, sp, fq, cls, ds, flags_at=None, lowdin_at=None, meta_flags_at=None):
    """A two-atom 'dimer' scan along x, with its E_int groups (monomer slices as terms)."""
    recs, groups = [], []
    for d in ds:
        rid = f"{name}:d{d:g}"
        lw = (lowdin_at or {}).get(d, [float(fq[0]), float(fq[1])])
        recs.append(_rec(rid, sp, [[0, 0, 0], [d, 0, 0]], [0, 1], fq, cls, d, (flags_at or {}).get(d, ()), lw))
        groups.append({"name": f"E_int:{rid}", "kind": "E_int", "ref": -0.001 * 14.4 * abs(fq[0] * fq[1] or 1) / d,
                       "terms": [[rid, 1.0], [f"{rid}#frag=0", -1.0], [f"{rid}#frag=1", -1.0]],
                       "meta": {"cls": cls, "dmin": d, "flags": list((meta_flags_at or {}).get(d, ()))}})
    return recs, groups


def _write_set(root, name, parts):
    recs = [r for p in parts for r in p[0]]
    groups = [g for p in parts for g in p[1]]
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{name}.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    with open(os.path.join(root, f"{name}.groups.json"), "w") as f:
        json.dump(groups, f)
    return recs, groups


def _result(setname, recs, groups, path, charges=True, fail=None, label="toy model"):
    """A minimal `lrbench eval --save-predictions` result: every term scored as 1.1 x its reference share."""
    P = {}
    for g in groups:
        for t, c in g["terms"]:
            P[t] = [0.0 if "#" in t else 1.1 * g["ref"]]
    if fail:
        P[fail] = [None]
    S = {"predictions": P}
    if charges:
        S["qfrag"] = {r["id"]: [[r["fragment_charges"][0] - 0.01, r["fragment_charges"][1] + 0.01]] for r in recs}
    return {"label": label, "level": {"model_level": LEVEL, "mismatch": False},
            "members": [{"external_model": {"long_range": "Coulomb", "charge_spin": "yes"}}],
            "data": {setname: {"jsonl": path}}, "sets": {setname: S}}


@pytest.fixture()
def world(tmp_path):
    data, runs, priv = tmp_path / "lr-sets", tmp_path / "runs", tmp_path / "internal"
    runs.mkdir()
    pub = _write_set(str(data), "ions", [
        _scan("Mg_H2O", ["Mg", "O"], [2, 0], "ion_polar", [3.0, 10.0, 20.0],
              flags_at={20.0: ["dft:pathology:ct_fractional"]}, lowdin_at={20.0: [1.2, -1.2 + 0.0]}),
        _scan("Mg_Cl", ["Mg", "Cl"], [2, -1], "ion_pair_ct_ambiguous", [4.0, 12.0]),
        _scan("Li_Na", ["Li", "Na"], [1, 1], "same_sign", [5.0, 10.0]),
        _scan("Na_Cl", ["Na", "Cl"], [1, -1], "ion_pair", [9.0, 12.0], meta_flags_at={12.0: ["ct_candidate"]}),
        _scan("Ne_Ne", ["Ne", "Ne"], [0, 0], "neutral", [5.0]),
    ])
    internal = _write_set(str(priv), "w18x", [_scan("Zn_H2O", ["Zn", "O"], [2, 0], "ion_polar", [4.0, 9.0])])
    J = _result("ions", *pub, path=str(data / "ions.jsonl"), fail="Li_Na:d5")
    (runs / "charged.json").write_text(json.dumps(J))
    J2 = _result("ions", *pub, path=str(data / "ions.jsonl"), charges=False, label="local model")
    (runs / "local.json").write_text(json.dumps(J2))
    # a result scored on the internal set, left in runs/ with the internal path recorded
    J3 = _result("w18x", *internal, path=str(priv / "w18x.jsonl"), label="internal run")
    (runs / "internal.json").write_text(json.dumps(J3))
    return {"data": data, "runs": runs, "priv": priv, "pub": pub, "J": J}


def test_rows_labels_and_values(world):
    recs, groups = world["pub"]
    S = world["J"]["sets"]["ions"]
    rows = [dict(zip(ionct.COLS, r)) for r in ionct.rows_for_set({r["id"]: r for r in recs}, groups, "ions",
                                                                  S["predictions"], S["qfrag"])]
    by = {r["id"]: r for r in rows}
    assert "Ne_Ne:d5" not in by                       # neutral pairs are not these panels' business
    assert by["Mg_H2O:d3"]["qA"] == 2 and by["Mg_H2O:d3"]["path"] == 0 and by["Mg_H2O:d3"]["phys"]
    flagged = by["Mg_H2O:d20"]
    assert flagged["path"] == 1 and flagged["ct"] and not flagged["phys"]          # kept, labelled, not scored
    assert by["Mg_Cl:d4"]["ct"] == ["class:ion_pair_ct_ambiguous"]
    assert by["Na_Cl:d12"]["ct"] == ["ct_candidate"] and by["Na_Cl:d9"]["ct"] == []   # group-meta flags count
    assert by["Li_Na:d5"]["model"] is None                                         # failed term: None, not zero
    assert by["Mg_Cl:d12"]["model"] == pytest.approx(1.1 * by["Mg_Cl:d12"]["ref"], rel=1e-6)
    assert by["Mg_Cl:d12"]["r"] == pytest.approx(12.0)
    assert by["Mg_H2O:d20"]["lwA"] == pytest.approx(1.2) and by["Mg_Cl:d4"]["qmA"] == pytest.approx(1.99)


def test_collect_attaches_points_and_charge_flags(world, monkeypatch):
    monkeypatch.setenv("LRBENCH_PRIVATE_IONS", str(world["priv"]))
    d = collect(str(world["runs"]), data=str(world["data"]))
    R = d["results"]
    assert R["charged"]["charges"] is True and R["local"]["charges"] is False      # per model, not per set
    assert len(R["charged"]["sets"]["ions"]["ionct"]["rows"]) == 9
    # the internal set is never read through collect(), so it cannot be frozen into a site
    assert "ionct" not in R["internal"]["sets"]["w18x"]
    assert "Zn_H2O" not in json.dumps(d)


def test_private_path_is_refused_even_as_the_data_dir(world, monkeypatch):
    monkeypatch.setenv("LRBENCH_PRIVATE_IONS", str(world["priv"]))
    d = collect(str(world["runs"]), data=str(world["priv"]))
    assert "Zn_H2O" not in json.dumps(d)


def test_private_points_reads_the_local_path(world):
    p = ionct.private_points(str(world["priv"]), str(world["runs"]))
    assert p["sets"] == ["w18x"] and len(p["ref"]["w18x"]) == 2
    assert "internal run" in p["models"]


def _load_tool():
    spec = importlib.util.spec_from_file_location("bds", ROOT / "tools" / "build_diagnostics_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_notebook_has_no_live_only_code(world, tmp_path, monkeypatch):
    monkeypatch.setenv("LRBENCH_PRIVATE_IONS", str(world["priv"]))
    data = collect(str(world["runs"]), data=str(world["data"]))
    out = _load_tool().freeze(NOTEBOOK, data, tmp_path / "frozen.py")
    src = out.read_text()
    assert "LIVE ONLY" not in src and "private_points" not in src
    assert not [line for line in src.splitlines() if "lrbench" in line.lower()]
    assert "Zn_H2O" not in src and "Mg_Cl" in src


def _run_notebook(path, monkeypatch, runs=None, data=None):
    pytest.importorskip("marimo")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    monkeypatch.setenv("LRBENCH_RUNS", str(runs) if runs else "/nonexistent-runs")
    if data:
        monkeypatch.setenv("LRBENCH_DATA", str(data))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location(f"nb_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    outs, defs = mod.app.run()
    html = "".join(str(getattr(o, "text", o)) for o in outs if o is not None)
    return html, defs


def test_notebook_panels_smoke(world, monkeypatch):
    monkeypatch.delenv("LRBENCH_PRIVATE_IONS", raising=False)
    html, defs = _run_notebook(NOTEBOOK, monkeypatch, runs=world["runs"], data=world["data"])
    assert len(defs["ION"]) > 0
    for s in ("Multivalent ions", "Charge transfer and leakage", "Leakage onto the neutral partner",
              "No charges", "local model", "No internal sets loaded"):
        assert s in html, s


def test_notebook_panels_with_internal_path(world, monkeypatch):
    monkeypatch.setenv("LRBENCH_PRIVATE_IONS", str(world["priv"]))
    html, defs = _run_notebook(NOTEBOOK, monkeypatch, runs=world["runs"], data=world["data"])
    assert defs["PRIV"]["sets"] == ["w18x"] and "Internal sets: interaction energy" in html


def test_notebook_cold(monkeypatch):
    monkeypatch.delenv("LRBENCH_PRIVATE_IONS", raising=False)
    html, defs = _run_notebook(NOTEBOOK, monkeypatch)
    assert defs["ION"].empty and "Multivalent ions" in html


def test_frozen_notebook_runs(world, tmp_path, monkeypatch):
    monkeypatch.setenv("LRBENCH_PRIVATE_IONS", str(world["priv"]))
    data = collect(str(world["runs"]), data=str(world["data"]))
    out = _load_tool().freeze(NOTEBOOK, data, tmp_path / "frozen.py")
    html, defs = _run_notebook(out, monkeypatch)
    assert len(defs["ION"]) > 0 and defs["PRIV"] == {}
    assert "never part of an exported notebook" in html
