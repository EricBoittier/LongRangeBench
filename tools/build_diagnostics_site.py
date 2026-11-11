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


def freeze(notebook: pathlib.Path, data: dict, out: pathlib.Path) -> pathlib.Path:
    """Write a copy of `notebook` with `data` substituted into its EMBEDDED literal."""
    lines = notebook.read_text().splitlines(keepends=True)
    i = next(n for n, line in enumerate(lines) if line.startswith(BEGIN))
    j = next(n for n, line in enumerate(lines) if line.startswith(END))
    frozen = f"    EMBEDDED = {json.dumps(json.dumps(data, separators=(',', ':')))}\n"
    out.write_text("".join(lines[:i + 1] + [frozen] + lines[j:]))
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
