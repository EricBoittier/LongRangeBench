# Handoff: what the v3 re-score at `bb3519d` still needs (2026-11-11)

This note is for whoever picks up the scoring batch. Sixteen models were re-scored on the seven canonical sets from
the pinned code commit `bb3519d`, in a local workspace (`runs-v3/`) that is not part of this repository. All sixteen
exited 0 and carry one clean commit with no `+dirty`. **Exit codes say nothing here**: four runs are contaminated by
a JAX GPU-memory fault that only the per-set `n_failed` counts reveal, and no row should be committed until the gate
in §3 passes.

Nothing in this note changes code. It records which rows are publishable, which must be re-scored, the root cause of
the contaminated runs, and the two traps that made the fault hard to see.

## 1. Status per model

| model | failures | cause | verdict |
|---|---|---|---|
| `dpa4-air`, `dpa4-plus`, `grace-2l`, `mace-h`, `mace-omol`, `mace-polar`, `mace-polar-l`, `orbmol`, `orbmol-v1`, `pet-mad` | 0 | — | publishable |
| `so3lr-tauri-s` | 0 | — | publishable (re-scored 22:08) |
| `oracle-refs` | 3361 | `outside the oracle's domain` | publishable; by design, the ceiling is defined only on far dimer points |
| `aimnet2` | 961 | `species not implemented` | publishable with a THIN COVERAGE flag; 18.6 % of 5176 structures, matching the earlier audit |
| `so3lr` (v1) | 1281 | **mixed** — see §2 | re-scoring in `v3c` |
| `so3lr-tauri` | 587 | JAX `RESOURCE_EXHAUSTED` | re-scoring in `v3c` |
| `so3lr-tauri-l` | 1200 | JAX `RESOURCE_EXHAUSTED` | re-scoring in `v3c` |

The `v3c` batch (`so3lr`, `so3lr-tauri`, `so3lr-tauri-l`, in that order, about an hour) started 22:31 with the fix in
§4. `so3lr-tauri-s` is not in it: it already has zero failures on all seven sets.

## 2. `so3lr` v1's 1281 failures are two different things

The count was read in-session as element coverage throughout. It is not. Per set, with the first recorded error:

| set | n_failed | first error | keep? |
|---|---|---|---|
| `dimers` | 1003 of 3762 | `unsupported element(s): K (Z=19)` | legitimate |
| `families_full` | 64 of 784 | `unsupported element(s): Na (Z=11)` | legitimate |
| `trimers` | 35 of 41 | `unsupported element(s): Li (Z=3)` | legitimate |
| `charge_parity` | 12 of 27 | `unsupported element(s): Li (Z=3)` | legitimate |
| `families_corrected` | 37 of 327 | `JaxRuntimeError: RESOURCE_EXHAUSTED … Failed to load in-memory CUBIN` | **contaminated** |
| `families_fixedq` | 130 of 230 | `JaxRuntimeError: RESOURCE_EXHAUSTED: Underlying backend ran out of memory … command buffer` | **contaminated** |

So 1114 are the model's declared element coverage (H, C, N, O, F, P, S, Cl) and 167 are the environment. The
element-coverage failures must survive the re-score unchanged; only the other two sets should move.

## 3. What is left to do

1. Let the `v3c` batch finish. One scoring process on the card at a time — a second one is how this started.
2. **Gate, before `lrbench results add`:** `n_failed == 0` on all seven sets for the three SO3LR-Tauri rows; for
   `so3lr` v1, `families_corrected` and `families_fixedq` must fall to its element coverage and neither set's
   first-error note may be a `JaxRuntimeError`. `aimnet2` and `oracle-refs` keep their counts by design.
3. Then `lrbench results add` for all sixteen, `lrbench results render`, and rebuild the explorer from the committed
   rows.

Keep every re-score on `bb3519d` so the rows stay comparable with the twelve that are already good.

## 4. Root cause: JAX preallocation starves XLA's own scratch

The failures are not about tensors. JAX preallocates a fraction of the card up front; XLA then needs memory *outside*
that pool to load CUBINs, resolve module functions and instantiate CUDA graphs. When the pool is too large, those
allocations fail while the model's own arrays fit — which is why the errors read `Failed to load in-memory CUBIN`,
`Failed to get module function`, and `ran out of memory trying to instantiate command buffer`, and why the run keeps
going and exits 0.

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

(`so3lr` v1's `dimers` and `families_full` failures are element coverage, not OOM; see §2. `trimers`,
`pathology_h2o2p` and `charge_parity` track `families_fixedq` in every row — near-total where it fails, zero where it
does not.)

Three things to read off this. The small set never fails and the large ones fail 73–100 %, so it is scratch, not
model capability. **The JAX default fraction was already too high on this card**: `so3lr` v1, `so3lr-tauri-l` and
even `so3lr-tauri-s` were contaminated in v3 with no override set. And `.92` was not uniformly worse than the
default — `so3lr-tauri-l` lost 72 dimer points at the default and none at `.92`, `so3lr-tauri-s` lost 12 at the
default and none at `.92` — so which structures get hit shifts from run to run. No preallocating setting was
reliable, and `so3lr-tauri-s` scoring clean once is not evidence that one was.

**The fix, verified:** `XLA_PYTHON_CLIENT_PREALLOCATE=false`, with no `MEM_FRACTION` set, so JAX allocates on demand
and leaves the rest of the card for XLA. Under it, the set that failed 230/230 completes 230/230. `v3c` uses this.

## 5. Two traps, for the next person reading these logs

- **`JaxRuntimeError` is in the result JSON's `notes`, not in the log.** The log holds metric rows; grepping it for
  `JaxRuntimeError` returns zero on a run whose JSON records six of them. `so3lr-tauri-l`'s re-score was called
  clean on exactly that mistake, twice, before its 1200 failures were read out of the JSON. Check `failures[*]
  .n_failed` and the `notes`; never the exit status, never a log grep.
- **Two `cuda_executor.cc:1182` allocation lines at startup are benign.** They are JAX's preallocation ladder
  stepping down to a size that fits, and a clean run has exactly these two and nothing else.

`unsupported element`, `species not implemented` and `outside the oracle's domain` are model and set properties. Keep
them, flag them, never re-score them away.

## 6. Provenance note

The first `so3lr-tauri` v3 result was overwritten by its own re-score at 21:46, so its original failure count cannot
be re-read; it was reported in-session as 362 and is unverifiable. Two surviving v3 results were copied to
`runs-v3/backup-v3/` before the running batch could overwrite them (`so3lr-tauri-s`, 12 failures, all OOM;
`so3lr-tauri-l`, 1427). An in-session summary table also mapped the three Tauri variants to the wrong counts — by
`created` stamp and per-set breakdown, `so3lr-tauri-l` was the 1427 and `so3lr-tauri-s` the 12, not the reverse. Trust the JSONs.
