#!/usr/bin/env python3
"""Freeze the current results into the diagnostics notebook and export it as a linked, self-contained site.

    python tools/build_diagnostics_site.py [RUNS_DIR] [-o OUT.html] [--mode run|edit] [--open]

The notebook's pages are `mo.routes`, so navigating between them runs Python -- which a static HTML export cannot
do (it renders no page at all). The export therefore goes through `marimo export html-wasm`, which runs the real
notebook in the browser via Pyodide. Pyodide has no access to the run directory, so the numbers are frozen into the
notebook's EMBEDDED literal first, by the same `lrbench.diagnostics.collect` the live notebook uses.

The result is one HTML file that needs no server, no Python and no checkout: open it, or put it behind any static
host. `--single-file` pulls the runtime from a CDN, so the first load needs internet access.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import webbrowser

HERE = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = HERE / "notebooks" / "long_range_diagnostics.py"
BEGIN = "    # --- BEGIN EMBEDDED DATA"
END = "    # --- END EMBEDDED DATA ---"


#: The loader used by a frozen notebook. It replaces the live one wholesale, so that a notebook carrying its data
#: contains **no mention of lrbench anywhere** — not an import, not a module name inside a string, not a sentence
#: in the docstring. marimo offers to install any package it believes a notebook wants, and a dead branch naming
#: an unpublished distribution is enough to make it offer; on a hosted runner that surfaces as
#: "Missing packages: lrbench", which is confusing and unfixable from the reader's side.
FROZEN_LOADER = '''@app.cell
def _(EMBEDDED):
    import io
    import json

    import marimo as mo
    import numpy as np
    import pandas as pd

    # This notebook carries its own data: everything below is read from the literal above, and nothing is imported
    # beyond the four packages declared in the script block at the top of the file.
    _d = json.loads(EMBEDDED)
    R, P = _d["results"], _d["priors"]
    SOURCE = _d.get("source", "frozen at build time")
    CHEM, VIEWS = _d.get("chemiscope", {}), _d.get("views", [])
    SCORES = _d.get("scores", {})

    KE = 14.399645            # e^2/(4 pi eps0), eV A - so k q_A q_B / r is in eV for r in A
    SET = "dimers"
    CLS = "ion_pair"
    DDE_SET = "families_fixedq"
    return CHEM, CLS, DDE_SET, KE, P, R, SCORES, SET, SOURCE, VIEWS, io, mo, np, pd


'''

LIVE_PROSE = ("numpy, pandas and matplotlib are in Pyodide, and nothing else is imported at the top level. "
              "`lrbench` is imported\nby name, and only when there is a filesystem with results to read.")
FROZEN_PROSE = ("numpy, pandas and matplotlib are in Pyodide, and this copy carries its own data, so nothing\n"
                "else is imported at all.")


def freeze(notebook: pathlib.Path, data: dict, out: pathlib.Path) -> pathlib.Path:
    """Write a copy of `notebook` with `data` in its EMBEDDED literal and the live loader replaced."""
    src = notebook.read_text()
    lines = src.splitlines(keepends=True)
    i = next(n for n, line in enumerate(lines) if line.startswith(BEGIN))
    j = next(n for n, line in enumerate(lines) if line.startswith(END))
    frozen = "    EMBEDDED = " + json.dumps(json.dumps(data, separators=(",", ":"))) + "\n"
    src = "".join(lines[:i + 1] + [frozen] + lines[j:])

    start = src.index("@app.cell\ndef _(EMBEDDED):")
    end = src.index("@app.cell", start + 10)
    src = src[:start] + FROZEN_LOADER + src[end:]
    src = src.replace(LIVE_PROSE, FROZEN_PROSE)
    src = src.replace("    marimo edit notebooks/long_range_diagnostics.py                  # live, reads "
                      "$LRBENCH_RUNS\n", "")
    src = src.replace("# The same system as the explorer page (lrbench.palette), with the values inlined because "
                      "this notebook also", "# The project's colour system, inlined because this notebook also")
    live_score = src[src.index("    def _score(name, r):"):src.index("    def _v(t, n=3):")]
    src = src.replace(live_score, "    def _score(name, r):\n        return SCORES.get(name, {})\n\n")

    leftover = [n for n, line in enumerate(src.splitlines(), 1) if "lrbench" in line.lower()]
    if leftover:
        print("[site] warning: 'lrbench' still appears on lines " + str(leftover[:6]) +
              " of the frozen notebook — a hosted runner may offer to install it")
    out.write_text(src)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default=os.environ.get("LRBENCH_RUNS", "runs"))
    ap.add_argument("-o", "--out", default="diagnostics-site.html")
    ap.add_argument("--mode", choices=("run", "edit"), default="run")
    ap.add_argument("--show-code", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the result in a browser when it is built")
    ap.add_argument("--also-static", action="store_true",
                    help="additionally write <out>-static.html: one scroll, no runtime needed")
    ap.add_argument("--data", default=None, help="dataset dir for the chemiscope views (default: RUNS/../lr-sets)")
    ap.add_argument("--no-chemiscope", action="store_true", help="skip the embedded chemiscope datasets")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(HERE / "src"))
    from lrbench.diagnostics import collect

    data = collect(a.runs)
    if not data["results"]:
        sys.exit(f"no results in {a.runs!r}: nothing to build")

    # chemiscope datasets travel with the notebook: Pyodide has no run directory, and the views are the point
    if not a.no_chemiscope:
        sys.path.insert(0, str(HERE / "tools"))
        try:
            from build_chemiscope_views import payloads, view_specs

            built = payloads(a.runs, a.data, verbose=True)
            data["chemiscope"] = {k: v["blob"] for k, v in built.items()}
            data["views"] = view_specs(built)
            print(f"[site] {len(data['chemiscope'])} chemiscope datasets, {len(data['views'])} views")
        except Exception as e:                       # chemiscope missing, or no saved predictions
            print(f"[site] no chemiscope views ({type(e).__name__}: {e})")
            data["chemiscope"], data["views"] = {}, []

    try:                              # precompute, so a frozen notebook needs no package to show the score
        from lrbench.diagnostics import score_components

        data["scores"] = {n: score_components(r) for n, r in data["results"].items()}
    except Exception as e:
        print(f"[site] no precomputed scores ({type(e).__name__}: {e})")
    data["source"] = f"{len(data['results'])} results, frozen at build time"   # not the local path

    size = len(json.dumps(data)) / 1024
    print(f"[site] {len(data['results'])} results, {len(data['priors'])} prior suites, {size:.0f} KB frozen")

    marimo = shutil.which("marimo") or str(pathlib.Path(sys.executable).parent / "marimo")
    out = pathlib.Path(a.out).resolve()
    tmp = out.parent / "diagnostics-frozen.py"
    freeze(NOTEBOOK, data, tmp)
    cmd = [marimo, "export", "html-wasm", str(tmp), "-o", str(out), "--single-file", "--mode", a.mode, "-f",
           "--show-code" if a.show_code else "--no-show-code"]
    print("[site]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        return r.returncode
    print(f"[site] wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("[site] self-contained: no server, no Python; the runtime loads from a CDN on first open")

    if a.also_static:
        # The notebook is one scroll, so a plain static export carries every section. It needs no Python runtime
        # at all: the figures are baked in and chemiscope runs as ordinary JS inside its iframe. What it loses is
        # the sliders, which need Python — that is what the WebAssembly build is for.
        static = out.with_name(out.stem + "-static" + out.suffix)
        r2 = subprocess.run([marimo, "export", "html", str(tmp), "-o", str(static), "-f"],
                            capture_output=True, text=True)
        if r2.returncode:
            sys.stderr.write(r2.stdout + r2.stderr)
        else:
            print(f"[site] wrote {static}  ({static.stat().st_size / 1024:.0f} KB, every page in one scroll)")
    # The frozen notebook is kept, not deleted: it is a self-contained marimo notebook with the data inside, which
    # is exactly what molab wants (open it from a GitHub URL, or upload it) and what `marimo edit` can reopen.
    print(f"[site] frozen notebook: {tmp}  ({tmp.stat().st_size / 1024:.0f} KB) — upload this to molab")
    if a.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
