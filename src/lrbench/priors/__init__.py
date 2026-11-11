"""Priors and assumptions suite (``lrbench priors``): modular physical-prior, model-assumption and data-assumption
checks, each with a pass/fail rule, for any model behind the lrbench backends. Separate from ``lrbench eval``,
whose outputs are unchanged."""
from . import checks  # noqa: F401  (registers the checks)
from .checks import ORDER
from .core import FAIL, NA, NE, PASS, REGISTRY, WARN, Ctx, run

__all__ = ["ORDER", "REGISTRY", "Ctx", "run", "PASS", "FAIL", "WARN", "NA", "NE"]
