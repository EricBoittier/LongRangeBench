#!/usr/bin/env python3
"""Freeze both notebooks with their data and export them to WebAssembly for the Pages site.

    python tools/build_notebook_site.py -o site            # site/notebook/ (diagnostics) and site/scorecard/

Pyodide has no filesystem with the data folder in it, so each notebook is first written as a frozen copy that
carries its data, then exported with `marimo export html-wasm`. Only files committed to the repository are read
(notebooks/data/lr_program/), so the Pages workflow can run this without access to the program's sources.

* diagnostics: tools/build_diagnostics_site.freeze() with the program payload (program.json + chemiscope.json). The
  benchmark sections need a run directory with predictions (LRBENCH_RUNS) and stay empty without one; the committed
  summary rows are on the results page.
* scorecard: lrb_data.py and every file of the data folder are inlined into the first cell (base64), and lrb_data
  reads them from memory (lrb_data.set_bundle).
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
NB = HERE / "notebooks"
DATA = NB / "data" / "lr_program"

SC_IMPORT = """    sys.path.insert(0, str(mo.notebook_dir()))
    import lrb_data as L
"""


def freeze_scorecard(out: pathlib.Path) -> pathlib.Path:
    src = (NB / "model_scorecard.py").read_text()
    assert src.count(SC_IMPORT) == 1, "scorecard import block changed; update tools/build_notebook_site.py"
    lrb = (NB / "lrb_data.py").read_text()
    files = {}
    for f in sorted(DATA.iterdir()):
        if not f.is_file():
            continue
        raw = f.read_bytes()
        if f.name == "program.json":          # the scorecard reads only the status, scan and pending blocks
            P = json.loads(raw)
            raw = json.dumps({k: P.get(k) for k in ("status", "scans", "pending", "ladder_large_n", "evidence_limits", "typeheld_pdiff_text")}, separators=(",", ":")).encode()
        elif f.name == "chemiscope.json":     # ... and only the 1-D scan viewers
            C = json.loads(raw)
            keep = [v for v in C["views"] if v["key"].startswith("lrp-scan-")]
            raw = json.dumps({"views": keep, "chemiscope": {v["key"]: C["chemiscope"][v["key"]] for v in keep}},
                             separators=(",", ":")).encode()
        files[f.name] = base64.b64encode(raw).decode()
    frozen = ("    # frozen copy: lrb_data and the data folder are inlined (tools/build_notebook_site.py)\n"
              "    import base64 as _b64\n    import json as _js\n    import types as _types\n"
              f"    _LRB_SRC = {json.dumps(lrb)}\n"
              f"    _BUNDLE = {json.dumps(json.dumps(files, separators=(',', ':')))}\n"
              "    L = _types.ModuleType(\"lrb_data\")\n"
              "    exec(compile(_LRB_SRC, \"lrb_data.py\", \"exec\"), L.__dict__)\n"
              "    L.set_bundle({_k: _b64.b64decode(_v) for _k, _v in _js.loads(_BUNDLE).items()})\n")
    out.write_text(src.replace(SC_IMPORT, frozen))
    return out


def freeze_diagnostics(out: pathlib.Path) -> pathlib.Path:
    sys.path.insert(0, str(HERE / "tools"))
    from build_diagnostics_site import freeze

    data = {"results": {}, "priors": {}, "scores": {}, "chemiscope": {}, "views": [],
            "source": "benchmark results are not embedded in this copy (see the results page); the PET-OMol-LR "
                      "program section carries its data"}
    lrp = {"program": json.loads((DATA / "program.json").read_text())}
    if (DATA / "chemiscope.json").exists():
        lrp.update(json.loads((DATA / "chemiscope.json").read_text()))
    data["lr_program"] = lrp
    return freeze(NB / "long_range_diagnostics.py", data, out)


def export(nb: pathlib.Path, dest: pathlib.Path, mode: str) -> None:
    marimo = shutil.which("marimo") or str(pathlib.Path(sys.executable).parent / "marimo")
    cmd = [marimo, "export", "html-wasm", str(nb), "-o", str(dest), "--mode", mode, "-f", "--no-show-code"]
    print("[site]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(r.returncode)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="site")
    ap.add_argument("--mode", choices=("run", "edit"), default="run")
    ap.add_argument("--keep-frozen", default=None, help="directory to keep the frozen notebooks in (molab upload)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(a.keep_frozen or (out / ".frozen"))
    tmp.mkdir(parents=True, exist_ok=True)
    d = freeze_diagnostics(tmp / "long_range_diagnostics.py")
    s = freeze_scorecard(tmp / "model_scorecard.py")
    for f in (d, s):
        print(f"[site] frozen {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    export(d, out / "notebook", a.mode)
    export(s, out / "scorecard", a.mode)
    if not a.keep_frozen:
        shutil.rmtree(tmp)
    for sub in ("notebook", "scorecard"):
        tot = sum(f.stat().st_size for f in (out / sub).rglob("*") if f.is_file())
        print(f"[site] {out / sub}: {tot / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
