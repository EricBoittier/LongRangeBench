"""The page is published as well as opened from disk, and those two destinations do not accept the same assets.

A Claude artifact's content security policy admits stylesheets only from the artifact's own files and Google Fonts,
so a KaTeX stylesheet loaded from a CDN is dropped there without comment and every equation renders as a heap of
unstyled spans. The builder therefore vendors KaTeX beside the page. These tests pin the mode contract and never
touch the network.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _mod():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("bcv", ROOT / "tools" / "build_chemiscope_views.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_cache(tmp_path):
    """A cache that looks complete, so nothing has to be downloaded."""
    c = tmp_path / "cache"
    (c / "fonts").mkdir(parents=True)
    (c / "katex.min.css").write_text("@font-face{src:url(fonts/KaTeX_Main-Regular.woff2) format(\"woff2\")}")
    for i in range(20):
        (c / "fonts" / f"KaTeX_F{i}-Regular.woff2").write_bytes(b"\x00")
    return c


def test_cdn_mode_never_vendors(tmp_path):
    m = _mod()
    href = m.install_katex(tmp_path / "page.html", mode="cdn", verbose=False)
    assert href == m.KATEX_CDN
    assert not (tmp_path / "katex").exists()


def test_vendor_mode_copies_beside_the_page_and_returns_a_relative_href(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "katex_assets", lambda *a, **k: _fake_cache(tmp_path))
    href = m.install_katex(tmp_path / "out" / "page.html", mode="vendor", verbose=False)
    assert href == "katex/katex.min.css", "a published page must reference its own copy, not an absolute URL"
    assert (tmp_path / "out" / "katex" / "katex.min.css").is_file()
    assert len(list((tmp_path / "out" / "katex" / "fonts").glob("*.woff2"))) == 20


def test_vendor_mode_fails_loudly_rather_than_shipping_a_page_that_loses_its_maths(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "katex_assets", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="KaTeX"):
        m.install_katex(tmp_path / "page.html", mode="vendor", verbose=False)


def test_auto_mode_falls_back_to_the_cdn_when_offline(tmp_path, monkeypatch):
    """A build with no network still has to produce a usable page; it just cannot produce a publishable one."""
    m = _mod()
    monkeypatch.setattr(m, "katex_assets", lambda *a, **k: None)
    assert m.install_katex(tmp_path / "page.html", mode="auto", verbose=False) == m.KATEX_CDN


def test_a_partial_fetch_is_not_mistaken_for_a_cached_copy(tmp_path):
    """The stylesheet is written last, so an interrupted download leaves the cache looking incomplete."""
    m = _mod()
    c = tmp_path / "cache"
    (c / "fonts").mkdir(parents=True)
    for i in range(5):                      # fonts present, css absent: not a usable cache
        (c / "fonts" / f"KaTeX_F{i}.woff2").write_bytes(b"\x00")
    assert m.katex_assets(c, fetch=False) is None
