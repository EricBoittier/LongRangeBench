"""The GGUF registry binds a model to the mlip.cpp build it was trained with. No binary, no weights: a fake binary
(a file with key strings in it and a COMMIT file beside it) and a patched metadata reader."""
import json
import os

import pytest

import lrbench.backends as B
import lrbench.gguf_registry as G


@pytest.fixture
def reg(tmp_path, monkeypatch):
    root, builds = tmp_path / "models", tmp_path / "builds"
    m = root / "M13"
    m.mkdir(parents=True)
    (m / "model.gguf").write_bytes(b"weights")
    (m / "q.gguf").write_bytes(b"charge head")
    for commit, keys in (("aaa1111", b"pet.plato_lr.lrc_qmodel pet.plato_lr.lrc_links"),
                         ("bbb2222", b"pet.plato_lr.lrc_qmodel")):
        d = builds / commit / "bin"
        d.mkdir(parents=True)
        (d / "mlipcpp-eval").write_bytes(b"\0" + keys + b"\0")
        (d / "COMMIT").write_text(commit + "\n")
    meta = {"general.name": "x", "pet.plato_lr.lrc_qmodel": "q.gguf", "pet.plato_lr.lrc_links": "backbone",
            "pet.composed_key": 1}
    monkeypatch.setattr(B, "gguf_metadata", lambda p: dict(meta))
    monkeypatch.setenv("LRBENCH_GGUF_ROOT", str(root))
    monkeypatch.setenv("LRBENCH_MLIPCPP_BUILDS", str(builds))
    monkeypatch.delenv(G.ENV_ALLOW, raising=False)
    e = G.entry_from_file("M13", str(m / "model.gguf"), "aaa1111", str(builds / "aaa1111/bin/mlipcpp-eval"),
                          title="test model", group="m", seed=13, level="L")
    R = {"schema": "lrbench-gguf-registry/1", "builds": {}, "models": {"M13": e}, "groups": {"m": ["M13"]}}
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(R))
    monkeypatch.setenv(G.ENV_REGISTRY, str(p))
    G._load.cache_clear()
    return {"root": root, "builds": builds, "entry": e, "model": str(m / "model.gguf")}


def test_entry_records_hashes_companions_and_unlisted_keys(reg):
    e = reg["entry"]
    assert e["gguf"] == "${LRBENCH_GGUF_ROOT}/M13/model.gguf"
    assert set(e["companions"]) == {"q.gguf"} and len(e["sha256"]) == 64
    assert e["keys_not_literal"] == ["pet.composed_key"]  # in no binary as a literal: not held against one


def test_name_resolves_and_build_is_chosen(reg):
    from lrbench.models import resolve
    assert resolve("m13") == reg["model"]
    assert G.expand("m") == ["M13"]
    b = G.backend_for(reg["model"], None)
    assert b["binary"].endswith("aaa1111/bin/mlipcpp-eval") and b["provenance"]["binary_check"]["ok"]


def test_wrong_build_is_refused_then_recorded(reg, monkeypatch):
    other = str(reg["builds"] / "bbb2222/bin/mlipcpp-eval")
    with pytest.raises(ValueError, match="lrc_links"):
        G.backend_for(reg["model"], other)
    monkeypatch.setenv(G.ENV_ALLOW, "1")
    chk = G.backend_for(reg["model"], other)["provenance"]["binary_check"]
    assert not chk["ok"] and chk["unknown_keys"] == ["pet.plato_lr.lrc_links"]


def test_missing_or_changed_companion_is_caught(reg):
    q = os.path.join(os.path.dirname(reg["model"]), "q.gguf")
    with open(q, "wb") as f:
        f.write(b"another charge head")
    with pytest.raises(ValueError, match="q.gguf sha256"):
        G.backend_for(reg["model"], None)
    os.remove(q)
    with pytest.raises(ValueError, match="q.gguf missing"):
        G.backend_for(reg["model"], None)


def test_unregistered_gguf_is_left_alone(reg, tmp_path):
    p = tmp_path / "other.gguf"
    p.write_bytes(b"x")
    assert G.backend_for(str(p), None) is None


def test_capi_library_of_the_build_is_chosen_and_checked(reg):
    lib = reg["builds"] / "aaa1111-capi" / "lib"
    lib.mkdir(parents=True)
    (lib / "libmlipcpp_c.so").write_bytes(b"\0pet.plato_lr.lrc_qmodel pet.plato_lr.lrc_links\0")
    (lib / "COMMIT").write_text("aaa1111\n")
    from lrbench.models import resolve
    assert resolve("mlipcpp-capi:m13") == "mlipcpp-capi:" + reg["model"]
    c = G.capi_for(reg["model"], None)
    assert c["lib"] == str(lib / "libmlipcpp_c.so") and c["provenance"]["binary_check"]["ok"]
    with pytest.raises(ValueError, match="needs aaa1111"):
        G.capi_for(reg["model"], str(reg["builds"] / "bbb2222/bin/mlipcpp-eval"))
