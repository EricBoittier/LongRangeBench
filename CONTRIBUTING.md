# Contributing

Thanks for looking. The two most useful contributions are **a model adapter** and **a dataset**, and both are small,
self-contained files. Adversarial reports about a metric are the third — see below.

## Development setup

```sh
git clone https://github.com/EricBoittier/LongRangeBench
cd LongRangeBench
pip install -e ".[dev,plot,viz]"
pytest                  # 33 tests, CPU only, ~15 s
ruff check .            # real defects only (F, E9); formatting is not enforced
```

The test suite needs no GPU, no download and no reference data: it builds the Lennard-Jones toy set
(`src/lrbench/toy.py`) whose references come from the calculator being scored, so an exact model must score zero
error, tail fraction 1 and ΔΔE slope 1, and a truncated one must score tail fraction 0. If you change a metric and
those assertions still pass, you have probably not broken the definition; if they fail, read the change again.

## Layout

```
src/lrbench/
  cli.py          every command and flag
  data.py         canonical records, groups, fragment slicing
  backends.py     mlip.cpp (GGUF), metatomic/TorchScript, ASE
  adapters.py     released external models as ASE calculators
  models.py       the named-model registry (name -> spec, level, env, licence)
  metrics.py      scorers per set; E_int / ΔE / ΔΔE / E3 / forces / charges
  normalise.py    physics-normalised errors (ε, k_BT, signed scales)
  stats.py        hierarchical bootstrap, conformal, CQR
  oracle.py       the multipole oracle (achievable ceiling)
  equivariance.py rotation / reflection / translation / permutation probes
  perf.py         timing protocol
  report.py       markdown / HTML tables
  viz.py          chemiscope export
  suite.py        multi-model runs, one environment each
  toy.py          the self-contained LJ set
tests/            the above, on the toy set
tools/            analysis and dataset-building scripts (tools/README.md)
docs/             metric definitions, datasets, models, schemas
```

## Adding a model

See [docs/adding-a-model.md](docs/adding-a-model.md). In short: anything with an ASE calculator can already be scored
with no code at all —

```sh
lrbench eval --model "ase:my_package:MyCalculator?cutoff=6.0" --sets dimers --data ./lr-sets
```

An adapter is worth adding when the model needs set-up (a checkpoint registry, x64 mode, a charge/spin convention) or
when it should appear in `lrbench models` and `lrbench suite`. A PR adds one function to `adapters.py`, one row to
`models.py` — including the **training level of theory** and the **licence of the weights** — and one test that the
adapter imports and builds.

## Adding a dataset

See [docs/adding-a-dataset.md](docs/adding-a-dataset.md). A set is records in the canonical schema plus group
definitions (which differences are scored). Requirements, and they are not negotiable because the metrics depend on
them:

- every record carries its **level of theory**; groups never mix levels
- differences are defined by **groups**, so per-element offsets and charge-state offsets cancel
- a **licence** that permits what you are asking us to do with the data
- known-bad references are **flagged**, not deleted, with `dft_pathology_do_not_score`

## Reporting a metric bug

If a diagnostic gives a model a score you can show is wrong — a model with a correct Coulomb tail scoring tail
fraction 0, or a flat model scoring ε 0 — please open an issue. Include the result JSON (it records the command, the
dataset hashes and the versions) and, if you can, a minimal case on the toy set. These reports matter more than
feature requests: the whole value of the benchmark is that its numbers mean what they claim.

## Changing a metric definition

Metric definitions are part of the scoring protocol, so changing one invalidates published numbers. A PR that changes
a definition should say so in the title, update [docs/metrics.md](docs/metrics.md) in the same commit, and explain in
the description what previously published results would change. Once protocol v1 is frozen
([ROADMAP.md](ROADMAP.md)), such changes bump the protocol version rather than editing it in place.

## Style

Concise, functional code: names that carry the meaning, comprehensions and small helpers where they stay readable,
docstrings where the behaviour is not obvious from context. Each module's docstring says what it computes and names
the definition it follows. Line length 120. Keep new dependencies out of the core — `numpy`, `scipy` and `ase` are
all a user needs to score a model; everything else is an extra.

## Code of conduct

Be decent. Scientific disagreement about a metric is welcome and should be argued with data; personal hostility is
not. Report problems to the repository owner.
