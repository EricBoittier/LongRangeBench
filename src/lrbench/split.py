"""Held-out-only scoring with a pair-level split (a JSON with ``heldout_point_ids`` per set).

``rescore(result, split)`` rebuilds each set's scorer from the predictions saved in a result JSON (``lrbench eval
--save-predictions``), keeping only records whose id is held out (plus every term of a group whose AB/first term is
held out, so groups stay whole) and only groups whose terms are all kept. Energy metrics only: forces and charges are
not saved in result JSONs, so force / charge-parity blocks are absent from held-out tables. Sets the split does not
cover (charge_parity, pathology_h2o2p) are dropped. The rescored result carries ``split`` = {file, side, n kept}."""
from __future__ import annotations

import copy
import json
from typing import Optional

from .data import DataSet, base_id, load_set


def filter_set(ds: DataSet, keep_ids: set) -> DataSet:
    groups = [g for g in ds.groups if base_id(g["terms"][0][0]) in keep_ids]
    ids = set(keep_ids) | {base_id(t) for g in groups for t, _ in g["terms"]}
    recs = [r for r in ds.records if r["id"] in ids]
    return DataSet(ds.name, recs, groups, dict(ds.source, split_filtered=True))


def unseen_ids(seen_path: str, data: str, sets, ignore_suffix: str = ":monomer") -> dict:
    """{set: ids} of records with no seen_by flag (flags ending in ':monomer' mark only a monomer slice and are ignored
    for the record itself); sets the seen file does not list are kept whole."""
    S = json.load(open(seen_path))["seen_by"]
    out = {}
    for s in sets:
        ds = load_set(s, data)
        flagged = {i for i, v in S.get(s, {}).items() if any(not x.endswith(ignore_suffix) for x in v)}
        out[s] = [r["id"] for r in ds.records if r["id"] not in flagged]
    return out


def rescore(R: dict, split_path: str, side: str = "heldout", data: Optional[str] = None, n_boot: int = 1000,
            ids: Optional[dict] = None) -> dict:
    from .metrics import Preds, SetScorer
    if ids is None:
        S = json.load(open(split_path))
        ids = S[f"{side}_point_ids"]
    else:
        S = {}
    out = copy.deepcopy({k: v for k, v in R.items() if k not in ("sets", "equivariance", "rotational_eint")})
    out["label"] = f"{R.get('label')} [{side}]"
    out["sets"], out["split"] = {}, {"file": split_path, "side": side, "seed": S.get("seed"), "n_kept": {}}
    norm = R.get("normalise")
    for s, blk in R.get("sets", {}).items():
        P = blk.get("predictions")
        if not P or s not in ids:
            continue
        src = data or (R.get("data", {}).get(s, {}) or {}).get("data")
        ds = filter_set(load_set(s, src), set(ids[s]))
        members = [{t: {"energy": v[m]} for t, v in P.items()} for m in range(len(next(iter(P.values()))))]
        out["sets"][s] = SetScorer(ds, Preds(members), n_boot=n_boot, normalise=norm).run()
        out["split"]["n_kept"][s] = len(ds.records)
    return out
