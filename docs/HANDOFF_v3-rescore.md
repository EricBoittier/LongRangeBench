# Handoff: the v3 re-score at `bb3519d` (2026-11-11)

This note is for whoever picks up the scoring batch. Sixteen models were re-scored on the seven canonical sets from
the pinned code commit `bb3519d`, in a local workspace (`runs-v3/`) that is not part of this repository. All sixteen
exited 0 and carry one clean commit with no `+dirty`. **Exit codes said nothing here**: four runs were contaminated
by a JAX GPU-memory fault that only the per-set `n_failed` counts revealed. The cause is understood (§4) and fixed;
fifteen of the sixteen rows are now sound, and the last is re-scoring. No row should be committed until the gate in
§3 passes.

Nothing in this note changes code. It records the state of each row, the root cause, and the three traps that made
the fault hard to see.

## 1. Status per model

| model | failures | cause | verdict |
|---|---|---|---|
| `dpa4-air`, `dpa4-plus`, `grace-2l`, `mace-h`, `mace-omol`, `mace-polar`, `mace-polar-l`, `orbmol`, `orbmol-v1`, `pet-mad` | 0 | — | publishable |
| `so3lr-tauri-s` | 0 | — | publishable (re-scored 22:08) |
| `so3lr-tauri` | 0 | — | publishable (re-scored 23:13, was 587) |
| `oracle-refs` | 3361 | `outside the oracle's domain` | publishable; by design, the ceiling is defined only on far dimer points |
| `aimnet2` | 961 | `species not implemented` | publishable with a THIN COVERAGE flag; 18.6 % of 5176 structures, matching the earlier audit |
| `so3lr` (v1) | 1112 | `unsupported element` only | publishable with a THIN COVERAGE flag (re-scored 22:51, was 1281 mixed — §2) |
| `so3lr-tauri-l` | 1200 | JAX `RESOURCE_EXHAUSTED` | **re-scoring**, started 23:13 |

## 2. `so3lr` v1's failures were two different things, and now are one

Its v3 count of 1281 was read in-session as element coverage throughout. It was not: two sets had failed on GPU
memory. The `v3c` re-score separates them cleanly.

| set | v3 | v3c | first error now |
|---|---|---|---|
| `dimers` | 1003 of 3762 | 1003 | `unsupported element(s): K (Z=19)` |
| `families_full` | 64 of 784 | 64 | `unsupported element(s): Na (Z=11)` |
| `charge_parity` | 12 of 27 | 12 | `unsupported element(s): Li (Z=3)` |
| `trimers` | 35 of 41 | **33** | `unsupported element(s): Li (Z=3)` |
| `families_corrected` | 37 of 327 | **0** | — |
| `families_fixedq` | 130 of 230 | **0** | — |

All 1112 that remain are the model's declared element coverage (H, C, N, O, F, P, S, Cl), and no `JaxRuntimeError`
is left in the result's `notes`. The row is sound and keeps a THIN COVERAGE flag.

Note `trimers`: 35 → 33. Two of those failures were GPU memory, not elements, even though the set's first recorded
error named an element. The split was 1112 / 169, not the 1114 / 167 the first-error notes implied — see the third
trap in §5.

## 3. What is left to do

1. Let `so3lr-tauri-l` finish (started 23:13, about 17 minutes).
2. **Gate, before `lrbench results add`:** `n_failed == 0` on all seven sets for the three SO3LR-Tauri rows — the
   first two already pass — and no `JaxRuntimeError` anywhere in a result's `notes`. `so3lr` v1 keeps its 1112
   element-coverage failures, `aimnet2` its 961 and `oracle-refs` its 3361, all by design.
3. Then `lrbench results add` for all sixteen, `lrbench results render`, and rebuild the explorer from the committed
   rows.

Keep every re-score on `bb3519d` so the rows stay comparable with the fifteen that are already good.

## 4. Root cause: JAX preallocation starves XLA's own scratch

The failures were not about tensors. JAX preallocates a fraction of the card up front; XLA then needs memory
*outside* that pool to load CUBINs, resolve module functions and instantiate CUDA graphs. When the pool is too large,
those allocations fail while the model's own arrays fit — which is why the errors read `Failed to load in-memory
CUBIN`, `Failed to get module function`, and `ran out of memory trying to instantiate command buffer`, and why the
run keeps going and exits 0.

The card is a 16,380 MiB RTX 4060 Ti that also carries the desktop compositor and an editor's GPU process, about
900 MiB together, so the margin is thin before anything is scored.

The fault scales with how much scratch a run needs, which is why it looked intermittent. Counting only the
OOM-caused failures, by run and set:

| run | preallocation | `dimers` 3762 | `families_full` 784 | `families_corrected` 327 | `families_fixedq` 230 |
|---|---|---|---|---|---|
| `so3lr` v1 (v3) | JAX default | — | — | 37 | 130 |
| `so3lr-tauri-l` (v3) | JAX default | 72 | 761 | 305 | 230 |
| `so3lr-tauri-l` (v3b) | `MEM_FRACTION=.92` | 0 | 574 | 326 | 230 |
| `so3lr-tauri` (v3b) | `MEM_FRACTION=.92` | 0 | 0 | 296 | 220 |
| `so3lr-tauri-s` (v3) | JAX default | 0 | 0 | 0 | 4 |
| `so3lr-tauri-s` (v3b) | `MEM_FRACTION=.92` | 0 | 0 | 0 | 0 |
| `so3lr` v1 (v3c) | **`PREALLOCATE=false`** | — | — | **0** | **0** |
| `so3lr-tauri` (v3c) | **`PREALLOCATE=false`** | **0** | **0** | **0** | **0** |

(`so3lr` v1's `dimers` and `families_full` failures are element coverage, not OOM; see §2. `trimers`,
`pathology_h2o2p` and `charge_parity` track `families_fixedq` in every row — near-total where it fails, zero where it
does not.)

Three things to read off this. The small set never fails and the large ones failed 73–100 %, so it is scratch, not
model capability. **The JAX default fraction was already too high on this card**: `so3lr` v1, `so3lr-tauri-l` and
even `so3lr-tauri-s` were contaminated in v3 with no override set. And `.92` was not uniformly worse than the
default — `so3lr-tauri-l` lost 72 dimer points at the default and none at `.92`, `so3lr-tauri-s` lost 12 at the
default and none at `.92` — so which structures get hit shifts from run to run. No preallocating setting was
reliable, and `so3lr-tauri-s` scoring clean once was not evidence that one was.

**The fix, verified:** `XLA_PYTHON_CLIENT_PREALLOCATE=false`, with no `MEM_FRACTION` set, so JAX allocates on demand
and leaves the rest of the card for XLA. Under it `so3lr-tauri` scored 0 failures on all 5176 structures, having
failed 587 an hour earlier, and `so3lr` v1's two contaminated sets went to 0 while its element-coverage failures
stayed put. The sets that failed 230/230 complete 230/230.

## 5. Three traps, for the next person reading these logs

- **`JaxRuntimeError` is in the result JSON's `notes`, not in the log.** The log holds metric rows; grepping it for
  `JaxRuntimeError` returns zero on a run whose JSON records six of them. `so3lr-tauri-l`'s re-score was called
  clean on exactly that mistake, twice, before its 1200 failures were read out of the JSON. Check `failures[*]
  .n_failed` and the `notes`; never the exit status, never a log grep.
- **Two `cuda_executor.cc:1182` allocation lines at startup are benign.** They are JAX's preallocation ladder
  stepping down to a size that fits, and a clean run has exactly these two and nothing else.
- **A set's `first error` note names one cause, so a mixed set reads as whichever failed first.** `so3lr` v1's
  `trimers` reported an unsupported element and lost two further structures to GPU memory behind it. Per-set
  attribution from the notes is a lower bound on contamination, not a split; only a clean re-score settles it.

`unsupported element`, `species not implemented` and `outside the oracle's domain` are model and set properties. Keep
them, flag them, never re-score them away.

## 6. Provenance note

The first `so3lr-tauri` v3 result was overwritten by its own re-score at 21:46, so its original failure count cannot
be re-read; it was reported in-session as 362 and is unverifiable. It has since been re-scored clean, so nothing
depends on it. Two surviving v3 results were copied to `runs-v3/backup-v3/` before the running batch could overwrite
them (`so3lr-tauri-s`, 12 failures, all OOM; `so3lr-tauri-l`, 1427). An in-session summary table also mapped the
three Tauri variants to the wrong counts — by `created` stamp and per-set breakdown, `so3lr-tauri-l` was the 1427 and
`so3lr-tauri-s` the 12, not the reverse. Trust the JSONs.
