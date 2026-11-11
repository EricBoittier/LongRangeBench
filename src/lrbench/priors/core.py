"""Check registry, outcomes and the evaluation context of the priors suite.

Outcomes: ``pass`` / ``fail`` (the rule was evaluated), ``warn`` (data assumptions: the result stands but needs the
stated caveat), ``not_applicable`` (the model or backend lacks the capability, e.g. no charges returned, no cell
support) and ``not_evaluable`` (no probe data, or every point below the numerical floor). Zero scorable points is
never a pass.
"""
from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..data import Structure

PASS, FAIL, WARN, NA, NE = "pass", "fail", "warn", "not_applicable", "not_evaluable"
FP32_REL = 1.2e-7  # ~2 ulp of a float32 total energy


def outcome(status: str, summary: str, **kw) -> dict:
    return dict(status=status, summary=summary, **kw)


@dataclass
class Check:
    name: str
    kind: str  # physical | model | data
    prior: str
    rule: str
    fn: Callable
    findings: str = ""  # which known defect it is built to catch


REGISTRY: Dict[str, Check] = {}


def check(name, kind, prior, rule, findings=""):
    def deco(fn):
        REGISTRY[name] = Check(name, kind, prior, rule, fn, findings)
        return fn
    return deco


def _key(s: Structure) -> str:
    h = hashlib.sha1()
    h.update(" ".join(s.species).encode())
    h.update(np.round(np.asarray(s.positions, float), 6).tobytes())
    h.update(f"{int(s.charge)}|{int(s.spin)}".encode())
    if s.cell is not None and any(s.pbc):
        h.update(np.round(np.asarray(s.cell, float), 6).tobytes())
    return h.hexdigest()


@dataclass
class Ctx:
    backend: object
    label: str = "model"
    dimers: object = None  # DataSet
    model_level: Optional[str] = None
    allow_level_mismatch: bool = False
    noise_rel: float = FP32_REL
    ipea: Optional[List[dict]] = None  # rows with structure + DFT IP/EA
    sepframes: Optional[List[dict]] = None
    seen: Optional[dict] = None  # {record_id: [sources]} for the dimer set
    train_coverage: Optional[List[dict]] = None
    ref_backend: object = None
    baseline: Optional[dict] = None  # an earlier priors result JSON (paired checks)
    cutoff: Optional[float] = None
    declared_local: Optional[bool] = None
    opts: dict = field(default_factory=dict)
    n_calls: int = 0
    _memo: Dict[str, dict] = field(default_factory=dict)

    def predict(self, structs: List[Structure]) -> List[dict]:
        """Memoised by species, positions (1e-6 A), charge, spin and cell. Checks that probe the backend's own
        state handling (caching, determinism) call ``backend.predict`` directly."""
        keys = [_key(s) for s in structs]
        todo, seen = [], set()
        for k, s in zip(keys, structs):
            if k not in self._memo and k not in seen:
                todo.append((k, s))
                seen.add(k)
        if todo:
            self.n_calls += len(todo)
            for (k, _), p in zip(todo, self.backend.predict([s for _, s in todo])):
                self._memo[k] = p
        return [self._memo[k] for k in keys]

    def energies(self, structs) -> np.ndarray:
        return np.array([np.nan if p.get("error") or p.get("energy") is None else p["energy"]
                         for p in self.predict(structs)])

    def noise(self, *E) -> float:
        """Numerical floor of a sum/difference of total energies (eV)."""
        return float(sum(abs(e) for e in E if np.isfinite(e)) * self.noise_rel)

    @property
    def conditioned(self) -> bool:
        return bool(getattr(self.backend, "passes_charge_spin", True))


def run(ctx: Ctx, names: List[str], log=print) -> dict:
    out = {}
    for n in names:
        c = REGISTRY[n]
        t0, c0 = time.time(), ctx.n_calls
        try:
            r = c.fn(ctx)
        except Exception as e:  # a check that crashes is reported, never silently passed
            r = outcome(NE, f"check raised {type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        r.update(kind=c.kind, prior=c.prior, rule=c.rule, findings=c.findings,
                 seconds=round(time.time() - t0, 2), model_calls=ctx.n_calls - c0)
        out[n] = r
        log(f"[priors] {n:28s} {r['status']:15s} {r['summary']}")
    return out
