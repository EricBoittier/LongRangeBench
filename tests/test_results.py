"""The committed-results layout: rows, set cards, the validator and the generated tables (docs/results.md).

Everything runs on sets built in tmp_path and scored with the exact Lennard-Jones calculator, so no number in this
file is a real result and nothing here needs a model download."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from conftest import LJ
from lrbench import results as RS
from lrbench.cli import main
from lrbench.toy import LEVEL, lj_calc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _eval(datadir, out, sets="toy"):
    main(["eval", "--model", LJ, "--sets", sets, "--data", str(datadir), "--no-hash", "--n-boot", "50",
          "--out", str(out)])
    return json.load(open(out))


def _as_released(R, monkeypatch, name="lj-exact"):
    """Make the LJ result look like a registry model's: a released name plus the package identity an adapter
    records. Only the identity is faked -- every number is the real LJ eval."""
    real = RS._registry_spec
    monkeypatch.setattr(RS, "_registry_spec", lambda n: LJ if n == name else real(n))
    R["label"] = name
    for m in R["members"]:
        m["external_model"] = {"package": "ase", "version": "3.0"}
    return R


# ------------------------------------------------------------------------------------------------ the repo itself

def test_committed_results_are_valid_and_rendered():
    """What is committed under results/ validates, and every generated table is current. If this fails after a
    registry or row change, run `lrbench results render` and commit the result."""
    root = os.path.join(REPO, "results")
    assert RS.check(root) == []
    assert RS.render(root, os.path.join(REPO, "docs", "models.md"), check_only=True) == []


@pytest.mark.skipif(shutil.which("git") is None or not os.path.isdir(os.path.join(REPO, ".git")),
                    reason="needs a git checkout")
def test_gitignore_keeps_private_results_out():
    def ignored(p):
        return subprocess.run(["git", "-C", REPO, "check-ignore", "-q", "--no-index", p]).returncode == 0
    assert ignored("results/private/toy/secret-model.json")
    assert ignored("results/private/sets.json")
    assert ignored("results/some-local-run.json")          # loose files at the top level stay local
    assert not ignored("results/toy/mace-polar.json")
    assert not ignored("results/sets.json") and not ignored("results/SUMMARY.md")


# ------------------------------------------------------------------------------------------------ scrubbing

def test_strip_dirs_removes_every_directory_but_leaves_urls():
    s = RS.strip_dirs
    assert s("/home/someone/lr-sets") == "lr-sets"
    assert s("ase:lrbench.adapters:metatomic?path='/scratch/u/ckpt/pet-omol-s.pt'") == \
        "ase:lrbench.adapters:metatomic?path='pet-omol-s.pt'"
    assert s("~/x/y.gguf") == "y.gguf" and s("$LRBENCH_UPET_DIR/pet-omol-s.pt") == "pet-omol-s.pt"
    assert s("runs/out.json") == "runs/out.json"          # relative: not a leak, left alone
    for kept in ("--model-level wB97M-V/def2-TZVPD", "force MAE 27.8 meV/A", "model PBE/PAW (OMat24) vs labels",
                 "(model - DFT)/DFT"):
        assert s(kept) == kept, "levels, units and ratios are not directories"
    assert s("see https://huggingface.co/lab-cosmo/upet") == "see https://huggingface.co/lab-cosmo/upet"
    assert s("lrbench.adapters:so3lr?model='so3lr'") == "lrbench.adapters:so3lr?model='so3lr'"


def test_eval_records_the_code_commit_and_hashes_adapter_checkpoints(datadir, tmp_path):
    R = _eval(datadir, tmp_path / "r.json")
    c = R["code_commit"]               # None outside this repo's own checkout; a sha (maybe +dirty) inside it
    assert c is None or __import__("re").match(r"^[0-9a-f]{40}(\+dirty)?$", c)
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    assert RS.spec_weights_file(f"ase:lrbench.adapters:metatomic?path='{w}'&name='x'") == str(w)
    assert RS.spec_weights_file("ase:lrbench.adapters:so3lr?model='so3lr'") == "ase:lrbench.adapters:so3lr?model='so3lr'"


# ------------------------------------------------------------------------------------------------ add / check / render

def test_add_check_render_round_trip(datadir, tmp_path, monkeypatch):
    root = str(tmp_path / "results")
    card = RS.add_set("toy", datadir, root, licence="CC0-1.0", public=True, description="LJ toy")
    assert card["reference_level"] == [LEVEL] and len(card["revisions"]) == 1
    R = _as_released(_eval(datadir, tmp_path / "r.json"), monkeypatch)
    rf = tmp_path / "released.json"
    json.dump(R, open(rf, "w"))

    [p] = RS.add(str(rf), root, commit=COMMIT)
    assert p == os.path.join(root, "toy", "lj-exact.json")
    row = json.load(open(p))
    assert set(row) == RS.ROW_KEYS and row["schema"] == RS.SCHEMA
    assert row["seeds"] == 1 and row["evidence"] == "screen" and row["n_failed"] == 0
    assert row["level_mismatch"] is False and row["code_commit"] == COMMIT
    assert row["data"]["jsonl_sha256"] == card["revisions"][0]["jsonl_sha256"]
    tail = {m["metric"]: m["value"] for m in row["metrics"]}
    assert tail["tail fraction, ion_pair (d >= 8 A)"].startswith("1.000")
    assert not any(str(tmp_path) in s for _, s in RS._walk_strings(row)), "a directory leaked into the row"
    assert RS.check(root) == []

    docs = tmp_path / "models.md"
    docs.write_text("x\n" + "\n".join(RS.BEGIN.format(b) + "\n" + RS.END.format(b) for b in ("models", "results")))
    assert set(RS.render(root, str(docs))) == {os.path.join(root, "SUMMARY.md"), str(docs)}
    assert RS.render(root, str(docs), check_only=True) == []
    md = open(os.path.join(root, "SUMMARY.md")).read()
    assert "## toy" in md and "`lj-exact`" in md and "screen (1 seed)" in md and "1.000" in md
    assert "`lj-exact` (screen)" in docs.read_text()

    # main() routes the same commands
    main(["results", "check", "--root", root])
    main(["validate", "--root", root])


def test_public_rows_refuse_what_cannot_be_published(datadir, tmp_path):
    """Unreleased (non-registry) models and unidentifiable weights never make a public row; private is the place."""
    root = str(tmp_path / "results")
    RS.add_set("toy", datadir, root, licence="CC0-1.0", public=True)
    rf = tmp_path / "r.json"
    _eval(datadir, rf)
    with pytest.raises(ValueError) as e:
        RS.add(str(rf), root, model="my-unreleased-ft", commit=COMMIT)
    assert "not a released model" in str(e.value) and "cannot be identified" in str(e.value)
    [p] = RS.add(str(rf), root, model="my-unreleased-ft", commit=COMMIT, private=True)
    assert p == os.path.join(root, "private", "toy", "my-unreleased-ft.json")
    assert RS.check(root) == [] and RS.check(root, private=True) == []
    assert "my-unreleased-ft" not in RS.summary_md(root)

    # without a recorded commit the add is refused unless one is given deliberately
    R = json.load(open(rf))
    R["code_commit"] = None
    json.dump(R, open(rf, "w"))
    with pytest.raises(ValueError, match="code_commit"):
        RS.add(str(rf), root, model="x", private=True)


def test_private_sets_never_get_public_rows(datadir, tmp_path, monkeypatch):
    root = str(tmp_path / "results")
    RS.add_set("toy", datadir, root, licence="internal only", public=False)
    rf = tmp_path / "r.json"
    json.dump(_as_released(_eval(datadir, tmp_path / "r0.json"), monkeypatch), open(rf, "w"))
    with pytest.raises(ValueError, match="no card"):
        RS.add(str(rf), root, commit=COMMIT)
    with pytest.raises(ValueError, match="already carded as private"):
        RS.add_set("toy", datadir, root, licence="CC0-1.0", public=True)


def test_validator_rules(datadir, tmp_path, monkeypatch):
    root = str(tmp_path / "results")
    RS.add_set("toy", datadir, root, licence="CC0-1.0", public=True)
    R = _as_released(_eval(datadir, tmp_path / "r.json"), monkeypatch)
    good = RS.row_from_result(R, "toy", commit=COMMIT)
    assert RS.validate_row(good, root) == []

    def errs(**kw):
        return " | ".join(RS.validate_row(dict(good, **kw), root))
    assert "level_mismatch is false" in errs(model_level="PBE/PAW (OMat24)")
    assert errs(model_level="PBE/PAW (OMat24)", level_mismatch=True) == ""
    assert "no note says what failed" in errs(n_failed=3)
    assert "evidence must be claim" in errs(seeds=4, members=good["members"] * 4)
    assert errs(seeds=4, evidence="claim", members=good["members"] * 4) == ""
    assert "+dirty" in errs(code_commit=COMMIT + "+dirty")
    assert "directory path" in errs(notes=["ran from /home/someone/runs"])
    assert "unknown keys" in errs(host="node01")
    assert "no metrics" in errs(metrics=[])
    monkeypatch.setattr(RS, "_registry_spec", lambda n: "ase:lrbench.adapters:mace_polar?model='polar-1-m'")
    assert "cannot relabel" in errs()
    assert "no revision" in errs(data=dict(good["data"], jsonl_sha256="0" * 64))


# ------------------------------------------------------------------------------------------------ a new set, as documented

def _doc_set(data_dir, scans):
    """Build a set exactly as docs/adding-a-dataset.md says: records + E_int groups through fragment slices."""
    from ase import Atoms
    os.makedirs(os.path.join(data_dir, "canonical"), exist_ok=True)

    def e(sp, x):
        a = Atoms(sp, positions=x)
        a.calc = lj_calc()
        return float(a.get_potential_energy()), a.get_forces().tolist()

    recs, groups = [], []
    for name, cls, tilt in scans:
        for d in [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]:
            sp = ["Ar", "Ar", "Ne", "Ne"]
            x = np.array([[0, 0, 0], [1.2, 0, 0], [d, tilt, 0], [d, tilt + 1.1, 0]])
            E, F = e(sp, x)
            ea, eb = e(sp[:2], x[:2])[0], e(sp[2:], x[2:])[0]
            rid = f"{name}:d{d}"
            recs.append(dict(id=rid, species=sp, positions=x.tolist(), cell=None, pbc=[False] * 3, charge=0, spin=1,
                             energy=E, forces=F, level=LEVEL, formal_charges=[0] * 4, fragment_id=[0, 0, 1, 1],
                             fragment_charges=[0, 0], fragment_spins=[1, 1], cls=cls, group=name, role="AB",
                             distance=d, flags=[], refs={}, extra={}))
            groups.append(dict(name=f"E_int:{rid}", kind="E_int", ref=E - ea - eb,
                               terms=[[rid, 1.0], [rid + "#frag=0", -1.0], [rid + "#frag=1", -1.0]],
                               meta=dict(cls=cls, dmin=d, pair=name)))
    with open(os.path.join(data_dir, "canonical", "myset.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(os.path.join(data_dir, "canonical", "myset.groups.json"), "w"))
    return str(data_dir)


def test_new_set_and_new_curves_path(tmp_path, monkeypatch):
    """docs/results.md 'Adding a set' and 'Adding curves': card it, score it, add the row; then add a scan, which is
    a new revision -- the old row stays valid but is shown stale, and a row on uncarded data is refused."""
    root = str(tmp_path / "results")
    v1 = _doc_set(tmp_path / "v1", [("ArNe:a", "ion_pair", 0.0), ("ArNe:b", "same_sign", 0.4)])
    RS.add_set("myset", v1, root, licence="CC0-1.0", public=True, note="initial")
    R1 = _as_released(_eval(v1, tmp_path / "r1.json", sets="myset"), monkeypatch)
    assert R1["sets"]["myset"]["E_int"]["tail"]["ion_pair"]["slope"][0] == pytest.approx(1.0), \
        "the exact calculator must follow the set's own references (the doc's sanity check)"
    json.dump(R1, open(tmp_path / "r1.json", "w"))
    RS.add(str(tmp_path / "r1.json"), root, commit=COMMIT)

    v2 = _doc_set(tmp_path / "v2", [("ArNe:a", "ion_pair", 0.0), ("ArNe:b", "same_sign", 0.4),
                                    ("ArNe:c", "ion_pair", 0.8)])
    R2 = _as_released(_eval(v2, tmp_path / "r2.json", sets="myset"), monkeypatch, name="lj-exact")
    json.dump(R2, open(tmp_path / "r2.json", "w"))
    with pytest.raises(ValueError, match="no revision on the set card"):
        RS.add(str(tmp_path / "r2.json"), root, model="lj-exact", commit=COMMIT)

    card = RS.add_set("myset", v2, root, licence="CC0-1.0", public=True, note="+1 ion-pair scan")
    assert [r["n_groups"] for r in card["revisions"]] == [12, 18]
    assert RS.check(root) == []
    assert "STALE" in RS.summary_md(root)
    RS.add(str(tmp_path / "r2.json"), root, commit=COMMIT)       # re-scored on the current revision
    assert "STALE" not in RS.summary_md(root)


def test_partial_failures_reach_the_row(datadir, tmp_path, monkeypatch):
    """A backend error and a silent non-finite energy both count, and the row says so in a note."""
    import lrbench.backends as B
    root = str(tmp_path / "results")
    RS.add_set("toy", datadir, root, licence="CC0-1.0", public=True)
    real = B.ASEBackend._predict

    for kind in ("error", "nan"):
        def flaky(self, structs, _k=kind):
            out = real(self, structs)
            for o in out[len(out) // 2:]:
                o.update(energy=None, forces=None, error="RuntimeError: boom") if _k == "error" else \
                    o.update(energy=float("nan"))
            return out
        monkeypatch.setattr(B.ASEBackend, "_predict", flaky)
        with pytest.warns(Warning):
            R = _eval(datadir, tmp_path / f"{kind}.json")
        row = RS.row_from_result(R, "toy", model="x", commit=COMMIT)
        assert row["n_failed"] > 0, kind
        assert any("fail" in n for n in row["notes"]), kind
        assert RS.validate_row(row, root, private=True) == [], kind
def test_a_thin_row_says_so_in_the_table():
    """"33 failed" reads as a footnote. "9 of ~42 scored (21%)" reads as what it is.

    PET-SPICE-S on the trimers is the real case: 9 groups scored, 33 failed, because the model does not support
    several elements and whole systems drop out. The old flag reported the failure count alone, which does not tell
    a reader that the metric beside it was computed on a fifth of the task -- and these failures are never a random
    sample, so the surviving fifth is not a smaller version of the whole.
    """
    from lrbench.results import MIN_COVERAGE, _flags

    def flags(n_scored, n_failed, mismatch=False):
        return _flags({"level_mismatch": mismatch, "model_level": "L", "n_failed": n_failed,
                       "n_scored": n_scored, "evidence": "screen", "seeds": 1, "data": None}, None)

    thin = flags(9, 33)
    assert "THIN COVERAGE" in thin and "9 of ~42" in thin and "21%" in thin

    # complete coverage says nothing about failures at all
    assert "failed" not in flags(1254, 0) and "THIN" not in flags(1254, 0)

    # a few failures are reported with their coverage, but not as a warning
    ok = flags(95, 5)
    assert "THIN" not in ok and "95%" in ok and "5 failed" in ok

    # the threshold is the documented one, tested at the boundary rather than assumed
    assert "THIN" not in flags(int(MIN_COVERAGE * 1000), 1000 - int(MIN_COVERAGE * 1000))
    assert "THIN" in flags(int(MIN_COVERAGE * 1000) - 20, 1000 - int(MIN_COVERAGE * 1000) + 20)


def test_a_set_that_only_reports_refusals_gets_no_row():
    """``pathology_h2o2p`` reports counts of records the harness refused to score, for every model.

    A row for it would put a model's name in a results table beside no result at all, and a reader scanning that
    table does not distinguish "flagged 8 records" from "scored 8". So such a set is skipped, not published.
    """
    from lrbench.results import scored_metrics

    pathology_only = {"metrics": [{"metric": "pathology-flagged (not scored)", "value": "8"}]}
    assert scored_metrics(pathology_only) == []

    # a set that scores no *energies* can still measure something, and that row is publishable
    forces_too = {"metrics": [{"metric": "force MAE (meV/A)", "value": "51.8"},
                              {"metric": "pathology-flagged (not scored)", "value": "4"}]}
    assert len(scored_metrics(forces_too)) == 1
    assert scored_metrics({"metrics": []}) == [] and scored_metrics({}) == []


def test_coverage_flag_does_not_blame_a_model_for_a_set_with_no_energies():
    """``n_scored`` counts energy groups. charge_parity has none by construction -- it measures forces -- so
    reporting "0 % coverage" there would describe the set and read as an accusation against the model."""
    from lrbench.results import _flags

    def flags(n_scored, n_failed):
        return _flags({"level_mismatch": False, "model_level": "L", "n_failed": n_failed, "n_scored": n_scored,
                       "evidence": "screen", "seeds": 1, "data": None}, None)

    no_energies = flags(0, 12)
    assert "no energy groups in this set" in no_energies and "0%" not in no_energies
    assert "THIN COVERAGE" not in no_energies

    # a real shortfall is still called one
    assert "THIN COVERAGE" in flags(9, 33)
