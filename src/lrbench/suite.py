"""``lrbench suite``: one ``lrbench eval`` per named model, each in its own Python environment, then one combined
table (``combined.md`` / ``combined.html`` / figures) over every model that ran plus any ``--include`` results.

Layout of ``--outdir``::

    suite.json          what was asked, per model: command, environment, status (ok | failed | skipped | planned)
    run_<model>.sh      the exact eval command (also used by suite.sbatch)
    suite.sbatch        Slurm array, one task per model (``--dry-run`` writes it without running anything)
    <model>.json        lrbench eval result;  <model>.log  its stdout/stderr
    combined.md / combined.html / figs/

A model whose environment cannot import its package is ``skipped`` with the reason, and the table is still written.
Models trained at another level than the labels (SO3LR: PBE0+MBD(-NL) vs our wB97M-V/def2-TZVPD) are run with
``--allow-level-mismatch``, so the mismatch is recorded in their result and flagged in the combined table.
"""
from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

from .models import ENVS, expand, lookup


def python_for(env: str, pythons: Dict[str, str]) -> str:
    return pythons.get(env) or os.environ.get(f"LRBENCH_PY_{env.upper().replace('-', '_')}") or sys.executable


def eval_command(name: str, py: str, outdir: str, sets: str, data: str, device: str, extra: List[str]) -> List[str]:
    m = lookup(name)
    cmd = [py, "-m", "lrbench", "eval", "--model", m.spec, "--label", m.name, "--sets", sets, "--data", data,
           "--device", device, "--out", os.path.join(outdir, f"{_fname(m.name)}.json")]
    if not m.same_level_as_ours and "--allow-level-mismatch" not in extra:
        cmd.append("--allow-level-mismatch")
    return cmd + list(extra)


def _fname(name: str) -> str:
    return name.replace(":", "_")


def probe(py: str, module: str) -> Optional[str]:
    """None if ``py`` can import ``module`` and lrbench, else the reason."""
    r = subprocess.run([py, "-c", f"import lrbench, {module}"], capture_output=True, text=True)
    if r.returncode == 0:
        return None
    tail = (r.stderr.strip().splitlines() or ["?"])[-1]
    return f"{py} cannot import {module} / lrbench: {tail}"


def write_sbatch(path: str, runs: List[dict], header: Optional[str]):
    lines = ["#!/bin/bash"]
    if header and os.path.exists(header):
        lines += [l.rstrip() for l in open(header) if l.startswith("#SBATCH")]
    else:
        lines += ["#SBATCH --job-name=lrbench-suite", "#SBATCH --gpus=1", "#SBATCH --time=04:00:00",
                  "# add --account / --partition for your cluster (or pass --sbatch-header)"]
    lines += [f"#SBATCH --array=0-{len(runs) - 1}", "set -euo pipefail", "unset NVIDIA_TF32_OVERRIDE",
              "RUNS=(" + " ".join(shlex.quote(r["script"]) for r in runs) + ")", 'bash "${RUNS[$SLURM_ARRAY_TASK_ID]}"',
              "# then: lrbench compare <outdir>/*.json --md combined.md   (or rerun lrbench suite without --dry-run)"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_suite(models: str, outdir: str, sets: str = "dimers,families_fixedq,trimers", data: str = "hf",
              device: str = "cuda", extra: Optional[List[str]] = None, pythons: Optional[Dict[str, str]] = None,
              include: Optional[List[str]] = None, dry_run: bool = False, sbatch_header: Optional[str] = None) -> dict:
    extra, pythons, include = list(extra or []), dict(pythons or {}), list(include or [])
    os.makedirs(outdir, exist_ok=True)
    S = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "models": expand(models), "sets": sets,
         "data": data, "device": device, "extra_eval_args": extra, "include": include, "runs": []}
    for name in S["models"]:
        m = lookup(name)
        py = python_for(m.env, pythons)
        cmd = eval_command(name, py, outdir, sets, data, device, extra)
        script = os.path.join(outdir, f"run_{_fname(name)}.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nset -euo pipefail\nunset NVIDIA_TF32_OVERRIDE\n" + shlex.join(cmd) + "\n")
        os.chmod(script, 0o755)
        run = {"name": name, "title": m.title, "env": m.env, "python": py, "level": m.level,
               "level_mismatch_allowed": "--allow-level-mismatch" in cmd, "command": cmd, "script": script,
               "result": cmd[cmd.index("--out") + 1], "log": os.path.join(outdir, f"{_fname(name)}.log")}
        S["runs"].append(run)
        if dry_run:
            run["status"] = "planned"
            continue
        why = probe(py, ENVS[m.env]["probe"])
        if why:
            run.update(status="skipped", reason=why)
            continue
        with open(run["log"], "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=_env()).returncode
        ok = rc == 0 and os.path.exists(run["result"])
        run.update(status="ok" if ok else "failed", returncode=rc)
        if not ok:
            run["reason"] = _tail(run["log"])
    write_sbatch(os.path.join(outdir, "suite.sbatch"), S["runs"], sbatch_header)
    results = [r["result"] for r in S["runs"] if r.get("status") == "ok"] + include
    if results and not dry_run:
        S["combined_md"] = os.path.join(outdir, "combined.md")
        S["combined_html"] = os.path.join(outdir, "combined.html")
        from .report import compare
        try:
            import matplotlib  # noqa: F401
            figdir = os.path.join(outdir, "figs")
        except ImportError:
            figdir = None
        compare(results, html=S["combined_html"], figdir=figdir, md_out=S["combined_md"])
        with open(S["combined_md"], "a") as f:
            f.write(_status_md(S))
    with open(os.path.join(outdir, "suite.json"), "w") as f:
        json.dump(S, f, indent=1)
    return S


def _status_md(S) -> str:
    out = ["", "### Models in this suite", "", "| model | level | env | status | note |", "|---|---|---|---|---|"]
    for r in S["runs"]:
        note = "level mismatch with the labels (allowed, recorded)" if r["level_mismatch_allowed"] else ""
        if r.get("reason"):
            note = (note + "; " if note else "") + r["reason"].replace("|", "/")
        out.append(f"| {r['name']} ({r['title']}) | {r['level']} | {r['env']} | {r['status']} | {note} |")
    for p in S["include"]:
        out.append(f"| {os.path.basename(p)} | (included result) | | ok | |")
    return "\n".join(out) + "\n"


def _env():
    e = dict(os.environ)
    e.pop("NVIDIA_TF32_OVERRIDE", None)
    return e


def _tail(path, n=3) -> str:
    try:
        return " / ".join(open(path).read().strip().splitlines()[-n:])
    except OSError:
        return "?"
