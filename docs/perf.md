# Timing

A long-range term that triples the cost of a step changes what you can simulate, so cost belongs in the same harness
as accuracy, measured the same way for every model.

```sh
lrbench perf --model model.gguf --device cuda --boxes scan --out perf.json
lrbench perf --model model.gguf --boxes water3k,hca --box water3k=water-M.extxyz --box hca=min.extxyz \
    --md-steps 100 --out perf.json
```

## Protocol

- **Force call**: the median of `--n` calls (default 100) after `--warmup` calls (default 20), with every atom moved
  by a random ~1e-3 Å before each call so nothing is served from a cache. Compilation is excluded by the warm-up,
  which matters for JIT models where the first call can be seconds.
- **MD step**: with `--md-steps`, an ASE `VelocityVerlet` at 0.5 fs, median step time after warm-up (C API and ASE
  backends). The force call and the MD step are reported separately because the gap between them is the harness
  overhead, which is sometimes the larger number.
- **Memory**: peak GPU memory from `nvidia-smi` sampled every 200 ms while the row runs. This is **whole-device**, so
  do not share the GPU during a run.
- Each row records its code path, precision and GPU model, so rows from different machines are not silently mixed.

Reported per row: ms/step, µs/atom/step, ns/day at 0.5 fs, peak memory.

## Boxes

| name | what |
|---|---|
| `scan` | generated water boxes of 1k / 10k / 50k / 100k atoms (1.0 g/cm³ on a grid, random orientations) |
| `water<N>` | a generated water box of N atoms |
| any name | a structure file you give with `--box NAME=FILE` |

Generated boxes are **not equilibrated** — they exist to measure cost as a function of size and should never be used
for anything physical.

## Fragment-charge term

`--frag` times the host-side fragment-charge Coulomb term separately (with and without formal charges, on open water
clusters of `--frag-sizes` atoms), which is how you find out whether a long-range correction costs anything
measurable at scale or disappears into the noise of the neural-network evaluation.

## Precision

fp32 by default (`GGML_CUDA_NO_TF32=1`); `--tf32` sets `MLIPCPP_TF32=1`. `NVIDIA_TF32_OVERRIDE` is removed from the
backend environment on every run. A timing comparison across precisions is only meaningful if each row says which it
used, and each row does.
