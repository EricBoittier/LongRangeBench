"""Named models and ``lrbench suite``: names resolve to adapter specs, the four external models get the right level
handling, and a CPU suite over stand-in models (LJ under two names, one with a missing package) writes one combined
table. No external model package is needed."""
import json
import os

import pytest

from conftest import LJ
from lrbench import models as M
from lrbench.cli import main
from lrbench.data import default_model_level
from lrbench.suite import eval_command, run_suite


def test_external_names_resolve():
    assert M.expand("external") == ["so3lr", "so3lr-tauri", "mace-h", "mace-polar"]
    assert M.expand("Tauri,MACE_POLAR,so3lr-t") == ["so3lr-tauri", "mace-polar"]
    assert M.expand("lr-comparison") == ["so3lr", "mace-polar", "mace-polar-l"]  # etc/analysis/lr_comparison.py
    assert M.resolve("mace-polar") == "ase:lrbench.adapters:mace_polar?model='polar-1-m'"
    assert M.resolve("so3lr") == "ase:lrbench.adapters:so3lr?model='so3lr'"
    assert M.resolve("mace-h:omat_pbe") == "ase:lrbench.adapters:mace_mh?head='omat_pbe'"
    for spec in ("x.gguf", "m.pt", LJ, "oracle:multipole?order=2"):
        assert M.resolve(spec) == spec
    with pytest.raises(ValueError, match="unknown model name"):
        M.expand("mace-x")


def test_levels_of_named_models():
    lv = {n: default_model_level(M.resolve(n)) for n in M.EXTERNAL}
    assert lv == {"so3lr": "PBE0+MBD", "so3lr-tauri": "PBE0+MBD-NL", "mace-h": "wB97M-V/def2-TZVPD",
                  "mace-polar": "wB97M-V/def2-TZVPD"}
    for n in M.MODELS:
        assert M.lookup(n).level == default_model_level(M.MODELS[n].spec), n


def test_level_mismatch_flag_only_for_so3lr(tmp_path):
    for n in M.EXTERNAL:
        cmd = eval_command(n, "python", str(tmp_path), "dimers", "hf", "cuda", [])
        assert ("--allow-level-mismatch" in cmd) == n.startswith("so3lr"), n
        assert cmd[cmd.index("--label") + 1] == n


def test_requirements_table_covers_envs():
    md = M.requirements_md(M.EXTERNAL)
    for n in M.EXTERNAL:
        assert f"`{n}`" in md
    assert "so3lr[cuda12]" in md and "mace-torch[polar]" in md


def test_dry_run_writes_plan(tmp_path):
    S = run_suite("external", str(tmp_path), dry_run=True, pythons={"so3lr": "/opt/so3lr/bin/python"})
    assert [r["status"] for r in S["runs"]] == ["planned"] * 4
    sb = open(tmp_path / "suite.sbatch").read()
    assert "--array=0-3" in sb and "unset NVIDIA_TF32_OVERRIDE" in sb
    assert open(tmp_path / "run_so3lr.sh").read().split("\n")[3].startswith("/opt/so3lr/bin/python -m lrbench eval")
    assert json.load(open(tmp_path / "suite.json"))["models"] == M.EXTERNAL


@pytest.fixture
def standins(monkeypatch):
    envs = dict(M.ENVS, toy={"probe": "ase", "python": "", "install": [], "notes": ""},
                missing={"probe": "no_such_package_xyz", "python": "", "install": [], "notes": ""})
    mods = dict(M.MODELS)
    for name, env in (("lj-a", "toy"), ("lj-b", "toy"), ("lj-missing", "missing")):
        mods[name] = M.NamedModel(name, LJ, name, M.OMOL, env, "-", "-", "-", "-")
    monkeypatch.setattr(M, "ENVS", envs)
    monkeypatch.setattr(M, "MODELS", mods)
    import lrbench.suite as su
    monkeypatch.setattr(su, "ENVS", envs)


def test_suite_runs_and_combines(standins, datadir, tmp_path):
    out = tmp_path / "suite"
    main(["suite", "--models", "lj-a,lj-b,lj-missing", "--sets", "toy", "--data", datadir, "--device", "cpu",
          "--outdir", str(out), "--", "--n-boot", "50", "--no-hash"])
    S = json.load(open(out / "suite.json"))
    st = {r["name"]: r["status"] for r in S["runs"]}
    assert st == {"lj-a": "ok", "lj-b": "ok", "lj-missing": "skipped"}, [r.get("reason") for r in S["runs"]]
    md = open(out / "combined.md").read()
    header = md.splitlines()[0]
    assert "lj-a" in header and "lj-b" in header
    assert "| lj-missing (lj-missing) |" in md and "skipped" in md
    assert os.path.exists(out / "combined.html")
    r = json.load(open(out / "lj-a.json"))
    assert r["label"] == "lj-a" and r["sets"]["toy"]


def test_cli_takes_names():
    import argparse
    from lrbench.cli import _resolve_names
    a = argparse.Namespace(model="tauri", label=None, ensemble="so3lr,x.gguf")
    _resolve_names(a)
    assert a.model == "ase:lrbench.adapters:so3lr?model='so3lr-tauri-m-2606'" and a.label == "tauri"
    assert a.ensemble == "ase:lrbench.adapters:so3lr?model='so3lr',x.gguf"


def test_long_range_flag_has_one_source_and_no_default():
    """Every registry model declares whether it has an explicit long-range term; GGUF members read it from the GGUF
    registry (derived from the file's keys); a member nobody can classify is None ("?"), never False ("no")."""
    assert set(M.HAS_LR) == set(M.MODELS), "every registry model needs a has_lr entry"
    yes = {n for n, v in M.HAS_LR.items() if v}
    assert {"mace-polar", "mace-polar-l", "so3lr", "so3lr-tauri", "aimnet2", "maceles-off", "orbmol"} <= yes
    assert not yes & {"mace-h", "mace-omol", "pet-omol-s", "pet-mad", "orbmol-v1", "dpa4-plus", "grace-2l"}
    # the adapter's prose agrees with the flag wherever the prose says "none"
    for n, m in M.MODELS.items():
        if m.long_range.lower().startswith("none"):
            assert not m.has_lr, n
    from lrbench.gguf_registry import registry
    R = registry()
    assert all(isinstance(e.get("has_lr"), bool) for e in R["models"].values())
    flag = M.long_range_flag
    assert flag({"gguf_registry": {"name": "FTq14"}, "spec": "x/model.gguf"}) is True
    assert flag({"gguf_registry": {"name": "FTB13"}, "spec": "x/model.gguf"}) is False
    assert flag({"gguf_registry": {"name": "pet-omol-s-gguf"}}) is False
    assert flag({"spec": M.MODELS["mace-polar"].spec}) is True
    assert flag({"spec": M.MODELS["mace-omol"].spec}) is False
    assert flag({"spec": "some/unregistered.gguf"}) is None
    assert flag({"spec": "x", "has_lr": False}) is False


def test_distil_does_not_default_lr_or_charge(tmp_path):
    from lrbench.diagnostics import distil
    d = distil("FTq14", {"members": [{"spec": "m.gguf", "gguf_registry": {"name": "FTq14"}, "charge_spin": True}]})
    assert d["has_lr"] is True and d["charge_spin"] == "total charge + spin"
    d = distil("x", {"members": [{"spec": "unknown.gguf"}]})
    assert d["has_lr"] is None and d["charge_spin"] == "?"
