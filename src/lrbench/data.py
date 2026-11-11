"""Canonical records (SCHEMA.md) and group definitions: loading from HF or a local directory, subsystem slicing.

A *term* is a structure id, optionally with a fragment selector: ``"<id>#frag=<k>"`` or ``"<id>#frags=<k1,k2>"``.
Every term becomes one :class:`Structure` that a backend evaluates once.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Canonical sets. Access is gated while the reference data is unpublished (docs/datasets.md); override to point
# at your own copy or at the public dataset once it is live.
HF_REPO = os.environ.get("LRBENCH_HF_REPO", "EricBoi/omol-longrange-tests")
ALL_SETS = ["dimers", "families_full", "families_corrected", "families_fixedq", "trimers", "pathology_h2o2p",
            "charge_parity"]
PATHOLOGY = "dft_pathology_do_not_score"
ENERGY_DO_NOT_SCORE = "energy_do_not_score"  # e.g. SO3LR's OMol25/OPoly26 subsets (mixed energy conventions)
OMOL_LEVEL = "wB97M-V/def2-TZVPD"
UNSCORABLE_LEVELS = {"mixed", "unknown", ""}


def record_level(r: dict) -> str:
    return str(r.get("level") or "unknown")


def default_model_level(spec: str) -> str:
    """Training level a model spec implies: our GGUFs / metatomic exports are OMol25 (wB97M-V/def2-TZVPD)."""
    if spec.startswith("ase:"):
        from .adapters import default_level  # external models with a known training level (adapters.LEVELS)
        return default_level(spec)
    return OMOL_LEVEL if spec.endswith((".gguf", ".pt")) else "unknown"


@dataclass
class Structure:
    key: str
    species: List[str]
    positions: np.ndarray
    charge: int
    spin: int
    cell: Optional[np.ndarray] = None
    pbc: Tuple[bool, bool, bool] = (False, False, False)
    formal_charges: Optional[np.ndarray] = None
    atom_index: Optional[np.ndarray] = None  # indices into the parent record (subsystems)

    def atoms(self, formal: bool = False):
        from ase import Atoms
        a = Atoms(self.species, positions=self.positions, cell=self.cell if self.cell is not None else None,
                  pbc=self.pbc if self.cell is not None else False)
        # metatomic-ase reads info["charge"] / info["spin_multiplicity"] (else "spin"); mlipcpp-eval the same keys;
        # fairchem-style calculators read info["spin"] (also 2S+1 in OMol25)
        a.info.update(charge=int(self.charge), spin_multiplicity=int(self.spin), spin=int(self.spin))
        if formal and self.formal_charges is not None:
            a.arrays["formal_charges"] = np.asarray(self.formal_charges, float)
        return a


@dataclass
class DataSet:
    name: str
    records: List[dict]
    groups: List[dict]
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        ids = [r["id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: duplicate record ids")
        self.by_id = {r["id"]: r for r in self.records}

    def flagged(self, rid: str) -> bool:
        return PATHOLOGY in (self.by_id[rid].get("flags") or [])

    def group_flagged(self, g: dict) -> bool:
        m = g.get("meta") or {}
        if PATHOLOGY in (m.get("flags") or []) or m.get("pathology"):
            return True
        return any(self.flagged(base_id(t)) for t, _ in g["terms"] if base_id(t) in self.by_id)

    def energy_ok(self, rid: str) -> bool:
        """The record's energy may be scored: a known single level and no energy_do_not_score flag."""
        r = self.by_id[rid]
        return ENERGY_DO_NOT_SCORE not in (r.get("flags") or []) and record_level(r) not in UNSCORABLE_LEVELS

    def group_level(self, g: dict) -> Optional[str]:
        """The single level of every term's record, or None (mixed levels or an unscorable energy: never pooled)."""
        ids = {base_id(t) for t, _ in g["terms"]}
        if not all(i in self.by_id and self.energy_ok(i) for i in ids):
            return None
        lv = {record_level(self.by_id[i]) for i in ids}
        return lv.pop() if len(lv) == 1 else None

    def levels(self) -> List[str]:
        """The label levels of this set: every record carrying an energy, and every record a group with a
        reference value (``ref``) is built from. The second half is what makes a reference-only set (the ``ext_*``
        converters store CCSD(T) interaction energies on the groups, not per-record energies) carry its level, so
        that scoring one cross-level is refused or flagged like any other set."""
        lv = {record_level(r) for r in self.records if r.get("energy") is not None}
        for g in self.groups:
            if g.get("ref") is None:
                continue
            for t, _ in g["terms"]:
                r = self.by_id.get(base_id(t))
                if r is not None:
                    lv.add(record_level(r))
        return sorted(lv)

    def partner_copy(self, rid: str) -> bool:
        """A copy of a record from another config, kept only so groups are self-contained (not per-record scored)."""
        return "partner_copy" in (self.by_id[rid].get("flags") or [])


def base_id(term: str) -> str:
    return term.split("#", 1)[0]


def parse_term(term: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
    if "#" not in term:
        return term, None
    rid, sel = term.split("#", 1)
    k, v = sel.split("=", 1)
    if k not in ("frag", "frags"):
        raise ValueError(f"unknown term selector {sel!r} in {term!r}")
    return rid, tuple(int(x) for x in v.split(",") if x != "")


def combine_spins(mults: Sequence[int]) -> int:
    """Multiplicity of a union of fragments: high-spin coupling of the unpaired electrons (1 + sum(m - 1))."""
    return 1 + sum(int(m) - 1 for m in mults)


def structure(rec: dict, frags: Optional[Tuple[int, ...]] = None, key: Optional[str] = None) -> Structure:
    sp, pos = rec["species"], np.asarray(rec["positions"], float)
    cell = None if rec.get("cell") is None else np.asarray(rec["cell"], float)
    pbc = tuple(bool(p) for p in (rec.get("pbc") or (False,) * 3))
    fq = None if rec.get("formal_charges") is None else np.asarray(rec["formal_charges"], float)
    if frags is None:
        return Structure(key or rec["id"], list(sp), pos, int(rec["charge"]), int(rec.get("spin") or 1), cell, pbc, fq)
    fid = rec.get("fragment_id")
    if fid is None:
        raise ValueError(f"{rec['id']}: term asks for fragments {frags} but the record has no fragment_id")
    fid = np.asarray(fid)
    idx = np.where(np.isin(fid, frags))[0]
    if not len(idx):
        raise ValueError(f"{rec['id']}: no atoms in fragments {frags}")
    qf = rec.get("fragment_charges")
    if qf is None:
        raise ValueError(f"{rec['id']}: fragment subsystem without fragment_charges")
    sf = rec.get("fragment_spins") or [1] * len(qf)
    q = int(sum(int(qf[k]) for k in frags))
    s = combine_spins([sf[k] for k in frags])
    sub_fq = None
    if fq is not None:
        sub_fq = fq[idx]
        if abs(sub_fq.sum() - q) > 1e-6:  # the unit-first-atom convention must survive the slice
            sub_fq = None
    return Structure(key or f"{rec['id']}#frags={','.join(map(str, frags))}", [sp[i] for i in idx], pos[idx], q, s,
                     cell, pbc, sub_fq, idx)


def term_structure(ds: DataSet, term: str) -> Structure:
    rid, frags = parse_term(term)
    if rid not in ds.by_id:
        raise KeyError(f"{ds.name}: group term {term!r} references an unknown id")
    return structure(ds.by_id[rid], frags, key=term)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _hf_get(fname: str, revision: Optional[str]) -> Optional[str]:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(HF_REPO, fname, repo_type="dataset", revision=revision)
    except Exception as e:  # EntryNotFoundError and friends
        if "404" in str(e) or "EntryNotFound" in type(e).__name__ or "not found" in str(e).lower():
            return None
        raise


def locate(name: str, data: str, revision: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """(jsonl path, groups path or None) for one config; data = 'hf' or a local directory."""
    if data == "hf":
        return _hf_get(f"canonical/{name}.jsonl", revision), _hf_get(f"canonical/{name}.groups.json", revision)
    for d in (os.path.join(data, "canonical"), data):
        p = os.path.join(d, f"{name}.jsonl")
        if os.path.exists(p):
            g = os.path.join(d, f"{name}.groups.json")
            return p, g if os.path.exists(g) else None
    return None, None


def load_set(name: str, data: str = "hf", revision: Optional[str] = None) -> DataSet:
    jp, gp = locate(name, data, revision)
    if jp is None:
        raise FileNotFoundError(f"set {name!r} not found in {data!r} (canonical/{name}.jsonl)")
    recs = [json.loads(l) for l in open(jp) if l.strip()]
    for r in recs:
        r.setdefault("set", name)
        r["flags"] = r.get("flags") or []
        r["refs"] = r.get("refs") or {}
        r["extra"] = r.get("extra") or {}
    groups = json.load(open(gp)) if gp else []
    src = {"data": data, "jsonl": jp, "jsonl_sha256": _sha256(jp), "groups": gp,
           "groups_sha256": _sha256(gp) if gp else None, "revision": revision}
    return DataSet(name, recs, groups, src)


def available_sets(data: str, revision: Optional[str] = None) -> List[str]:
    if data == "hf":
        from huggingface_hub import list_repo_files
        fs = list_repo_files(HF_REPO, repo_type="dataset", revision=revision)
        return [f[len("canonical/"):-len(".jsonl")] for f in fs if f.startswith("canonical/") and f.endswith(".jsonl")]
    out = []
    for d in (os.path.join(data, "canonical"), data):
        if os.path.isdir(d):
            out += [f[:-6] for f in sorted(os.listdir(d)) if f.endswith(".jsonl")]
    return sorted(set(out))


def subsample(ds: DataSet, n: Optional[int], seed: int = 0) -> DataSet:
    """Keep about n records, whole scans/groups at a time (record.group, else the id), so differences stay intact."""
    if not n or len(ds.records) <= n:
        return ds
    by: Dict[str, List[dict]] = {}
    for r in ds.records:
        by.setdefault(r.get("group") or r["id"], []).append(r)
    keys = sorted(by)
    random.Random(seed).shuffle(keys)
    keep: List[dict] = []
    for k in keys:
        if len(keep) >= n:
            break
        keep += by[k]
    ids = {r["id"] for r in keep}
    # pull in records referenced by groups that are partially kept, so no group loses a term
    groups = [g for g in ds.groups if any(base_id(t) in ids for t, _ in g["terms"])]
    for g in groups:
        ids |= {base_id(t) for t, _ in g["terms"]}
    recs = [r for r in ds.records if r["id"] in ids]
    groups = [g for g in ds.groups if all(base_id(t) in ids for t, _ in g["terms"])]
    src = dict(ds.source, subsample=n, subsample_seed=seed, n_records_full=len(ds.records))
    return DataSet(ds.name, recs, groups, src)


def needed_terms(ds: DataSet) -> List[str]:
    """Every record id (absolute energies/forces) plus every group term, in a stable order."""
    seen, out = set(), []
    for t in [r["id"] for r in ds.records] + [t for g in ds.groups for t, _ in g["terms"]]:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
