# Adding a model

## You may not need to add anything

If your model has an ASE calculator, it can already be scored, with no code in this repository:

```sh
lrbench eval --model "ase:my_package.calculators:MyPotential?cutoff=6.0&device='cuda'" \
    --sets dimers,families_fixedq --data ./lr-sets --normalise --out mine.json
```

The spec is `ase:<module>:<factory>?<kwargs>`. The factory is any callable returning a calculator; keyword values are
parsed as Python literals, so quote strings. Two things to get right:

- **Total charge and multiplicity.** LongRangeBench puts them in `atoms.info["charge"]` and
  `atoms.info["spin_multiplicity"]`. If your calculator reads different keys, you need the thin adapter below.
- **Per-atom charges and dipole.** Optional, but if your calculator puts `charges` (and `dipole`) in
  `self.results`, the charge-parity and dipole blocks are scored too.

Start with the toy set — it takes seconds and tells you immediately whether the plumbing works:

```sh
lrbench toy ./toy-data
lrbench eval --model "ase:my_package:MyPotential" --sets toy --data ./toy-data --normalise --out mine.json
```

## When to write an adapter

Add one to `src/lrbench/adapters.py` when the model needs set-up that a user should not have to remember: a
checkpoint registry, float64 mode, a cutoff that must be disabled for vacuum, a non-standard charge/spin convention,
or a result key that needs unpacking. The `Adapter` class does the mechanical parts — info-key mapping, charge and
dipole extraction, metadata — so a new one is usually ten lines:

```python
def my_model(model: str = "my-1-m", dtype: str = "float64", device: str = "cuda"):
    """MY-MODEL-1 (my-package >= 1.2; arXiv:XXXX.XXXXX): trained on <dataset> at <level>; <what makes it
    long-range>."""
    from my_package.calculators import my_calculator
    calc = my_calculator(model=model, device=device, default_dtype=dtype)
    meta = {"name": f"MY-MODEL-1 ({model})", "version": my_package.__version__,
            "licence": "<licence of the weights>",
            "charge_spin": "atoms.info charge / spin (2S+1)",
            "long_range": "<explicit Coulomb term? learned charges? nothing?>"}
    return Adapter(calc, MY_LEVEL, meta,
                   info_map={"spin_multiplicity": "spin"},            # our key -> the model's key
                   charges=lambda R: R.get("charges"),                 # from calc.results, or None
                   dipole=lambda R: R.get("dipole"))
```

Then:

1. **Register the level** in `LEVELS` so `--model-level` can be inferred without building the model. A wrong level is
   worse than no level: it lets the harness compare energies that are not comparable.
2. **Set `charge_spin=False`** if the model genuinely cannot see a total charge. The result then records that, and
   nobody reads its ion-pair numbers as though it had been told the charge.
3. **Add a row to `models.py`** if the model is publicly released, giving its name, spec, training level, environment,
   how to obtain the checkpoint, the **licence of the weights**, how charge and spin enter, and what its long-range
   term actually is. That row is what `lrbench models` and `lrbench suite` use.
4. **Add a test** that the adapter imports and builds — see `tests/test_adapters.py`. The test must not need the
   checkpoint: skip when the package is missing.

## Environments

A model family does not have to be installable next to LongRangeBench. `lrbench suite` runs each model with its own
Python interpreter (`LRBENCH_PY_<ENV>`, see `configs/external_envs.sh`), so JAX and PyTorch models can be compared in
one table without sharing an environment. If your model needs a new environment, add it to `ENVS` in `models.py` with
the Python version, the install command and any trap worth recording (a bad wheel, a cutoff default, a cache that
must be pre-fetched on a login node).

## What a good submission looks like

The numbers are only comparable if the run is honest about three things, all of which the result JSON already
records; please check them before publishing:

- the **level of theory** the model was trained at, and whether it matches the set's references;
- whether the model was given the **total charge** and formal charges, or could not accept them;
- what of the benchmark's data was in the model's **training set** — see the contamination flags in
  [datasets.md](datasets.md) and the held-out-only rescoring in `lrbench.split`.

A model that cannot take a charge, evaluated on ion pairs, is a perfectly publishable result — it just has to say so.
