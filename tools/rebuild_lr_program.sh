#!/usr/bin/env bash
# Rebuild the notebooks' data folder (notebooks/data/lr_program) from the program's score files, so that results that
# have landed since the last build (the AL and PAL four-seed repeats, ...) replace their 'not yet scored' entries.
#
#   bash tools/rebuild_lr_program.sh            # rebuild, rewrite to public labels, gate, summary
#   bash tools/rebuild_lr_program.sh --site     # also freeze both notebooks and export the WebAssembly site to site/
#
# Needs the private sources (LRNB_DATA_ROOT, default ~/data) and the private label map (LRP_PUBLIC_MAP); a Python with
# numpy, ase, scipy and chemiscope (PY, default: python3); LRBENCH_DATA = the canonical lrbench sets (*.jsonl) for the
# reference-only structure viewers. Exits non-zero if the public-label gate finds anything.
set -euo pipefail
cd "$(dirname "$0")/.."
# local defaults (interpreter, data roots) may live in a private env file next to the label map
ENV=${LRP_ENV:-${LRNB_DATA_ROOT:-$HOME/data}/lr-program/notebook_publish/rebuild.env}
[[ -f "$ENV" ]] && source "$ENV"
PY=${PY:-python3}
SITE=0; ARGS=()
for a in "$@"; do [[ "$a" == "--site" ]] && SITE=1 || ARGS+=("$a"); done
"$PY" tools/build_lr_program_data.py "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | { grep -v '^\[search\]\|^\[prog\] [A-Za-z0-9_]*: val500' || true; }
"$PY" tools/lrp_public.py --check notebooks/data/lr_program
"$PY" - <<'EOF'
import json, pathlib
d = pathlib.Path("notebooks/data/lr_program")
p = json.loads((d / "program.json").read_text())
print(f"[rebuild] built {p['status']['built']}: {len(p['rows'])} program rows")
for r in p["status"]["running"]:
    print(f"[rebuild]   {r['name']}: {r['state']}" + (f"; not yet scored: {', '.join(r['pending'])}" if r["pending"] else ""))
tot = sum(f.stat().st_size for f in d.iterdir())
big = max(d.iterdir(), key=lambda f: f.stat().st_size)
print(f"[rebuild] data folder {tot / 1e6:.1f} MB; largest file {big.name} {big.stat().st_size / 1e6:.1f} MB")
EOF
if [[ $SITE == 1 ]]; then
  "$PY" tools/build_notebook_site.py -o site
fi
