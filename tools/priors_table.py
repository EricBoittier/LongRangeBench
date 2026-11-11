#!/usr/bin/env python3
"""Check x model table (markdown) from `lrbench priors` result JSONs; later JSONs given with --override replace single
checks (e.g. a fragment_charges run through mlipcpp-eval for the charges the C API does not return).

    python priors_table.py OUT.md A.json B.json ... [--override A.json=charges_a.json ...]"""
import json
import sys

S = {"pass": "pass", "fail": "**fail**", "warn": "warn", "not_applicable": "n/a", "not_evaluable": "n/e"}


def main(argv):
    out, rest = argv[0], argv[1:]
    files = [a for a in rest if "=" not in a and not a.startswith("--")]
    over = dict(a.split("=", 1) for a in rest if "=" in a)
    R = []
    for f in files:
        r = json.load(open(f))
        if f in over:
            for k, v in json.load(open(over[f]))["checks"].items():
                v["summary"] = v["summary"] + " (via mlipcpp-eval)"
                r["checks"][k] = v
        R.append(r)
    names = list(R[0]["checks"])
    L = ["| check | kind | " + " | ".join(r["label"] for r in R) + " |", "|---|---|" + "---|" * len(R)]
    for n in names:
        L.append(f"| {n} | {R[0]['checks'][n]['kind']} | " + " | ".join(
            S.get(r["checks"].get(n, {}).get("status"), "-") for r in R) + " |")
    L += ["", "| check | " + " | ".join(r["label"] for r in R) + " |", "|---|" + "---|" * len(R)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(
            (r["checks"].get(n, {}).get("summary") or "").replace("|", "/") for r in R) + " |")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
