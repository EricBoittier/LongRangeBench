#!/usr/bin/env python3
"""Transpose selected rows of an `lrbench compare` markdown table into a compact models-as-rows table (md + html).

    python headline_table.py COMBINED.md OUT_STEM [--note TEXT]"""
import html
import re
import sys

ROWS = [("dimers: normalised tail eps, ion_pair", "ε ion pair"), ("dimers: normalised tail eps, same_sign", "ε like charge"),
        ("dimers: tail fraction, ion_pair", "tail frac. ion pair"), ("dimers: tail fraction, ion_polar", "tail frac. ion+polar"),
        ("dimers: offset-free dE error at 8 A, ion_pair", "offset-free 8 Å ion pair (meV)"),
        ("dimers: |E_int| error >= 12 A, ion_pair", "abs E_int err ≥ 12 Å ion pair (meV)"),
        ("families_fixedq: fixed-Q wire ddE slope", "wire ΔΔE slope"), ("trimers: E3 MAE", "E3 MAE (meV)"),
        ("trimers: dE3 vs far geometry MAE", "ΔE3 vs far MAE (meV)"),
        ("rotational spread of offset-free tail E_int(8 A)", "rot. tail sd 8 Å: median, p90 (meV)"),
        ("equivariance rotation", "rotation max abs ΔE (meV), max abs ΔF (meV/Å)"),
        ("emulation score", "emulation score, E_int MAE incl. DFT pathologies (meV)"),
        ("physics score", "physics score, pathologies excluded (meV)")]


def main(md, stem, *a):
    L = [l for l in open(md) if l.startswith("|")]
    head = [c.strip() for c in L[0].strip().strip("|").split("|")]
    models = head[1:-1]
    rows = {}
    for l in L[2:]:
        c = [x.strip().replace("\\|", "|") for x in re.split(r"(?<!\\)\|", l.strip()[1:-1])]
        rows.setdefault(c[0], c[1:-1])
    get = lambda k: next((v for r, v in rows.items() if r.startswith(k)), None)
    cols = [(t, get(k)) for k, t in ROWS]
    cols = [(t, v) for t, v in cols if v]
    out = ["| model | " + " | ".join(t for t, _ in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    out += ["| " + m + " | " + " | ".join(v[i] for _, v in cols) + " |" for i, m in enumerate(models)]
    note = a[a.index("--note") + 1] if "--note" in a else ""
    out += ["", note]
    open(f"{stem}.md", "w").write("\n".join(out) + "\n")
    th = "".join(f"<th>{html.escape(t)}</th>" for t, _ in cols)
    tr = "".join("<tr><td>" + html.escape(m) + "</td>" + "".join(f"<td>{html.escape(v[i])}</td>" for _, v in cols) + "</tr>"
                 for i, m in enumerate(models))
    open(f"{stem}.html", "w").write(
        "<!doctype html><html><head><meta charset='utf-8'><title>lrbench headline</title><style>body{font-family:Arial,"
        "Helvetica,sans-serif;font-size:12px;margin:16px;background:#fff;color:#1a1a1a}table{border-collapse:collapse}"
        "td,th{border-bottom:1px solid #ddd;padding:3px 6px;text-align:left;vertical-align:top}</style></head><body>"
        f"<table><tr><th>model</th>{th}</tr>{tr}</table><p>{html.escape(note)}</p></body></html>")
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
