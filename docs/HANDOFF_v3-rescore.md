# Handoff: what the v3 re-score at `bb3519d` still needs (2026-11-11)

This note is for whoever picks up the scoring batch. Sixteen models were re-scored on the seven canonical sets from
the pinned code commit `bb3519d`, in a local workspace (`runs-v3/`) that is not part of this repository. All sixteen
exited 0 and carry one clean commit with no `+dirty`. **Exit codes say nothing here**: three runs are contaminated by
GPU memory faults that only the per-set `n_failed` counts reveal, and no row should be committed until the gate in
§3 passes.

Nothing in this note changes code. It records which rows are publishable, which must be re-scored, and the two
environment faults that produced failures invisible to the exit status.

## 1. Status per model

| model | failures | cause | verdict |
|---|---|---|---|
| `dpa4-air`, `dpa4-plus`, `grace-2l`, `mace-h`, `mace-omol`, `mace-polar`, `mace-polar-l`, `orbmol`, `orbmol-v1`, `pet-mad` | 0 | — | publishable |
| `so3lr-tauri-s` | 0 | — | publishable (re-scored 22:08) |
| `oracle-refs` | 3361 | `outside the oracle's domain` | publishable; by design, the ceiling is defined only on far dimer points |
| `aimnet2` | 961 | `species not implemented` | publishable with a THIN COVERAGE flag; 18.6 % of 5176 structures, matching the earlier audit |
| `so3lr` (v1) | 1281 | **mixed** — see §2 | **re-score two sets** |
| `so3lr-tauri` | 587 | JAX `RESOURCE_EXHAUSTED` | **re-score**, alone on the card |
| `so3lr-tauri-l` | 1427 (v3) | JAX `RESOURCE_EXHAUSTED` | re-score in progress since 22:08, clean through `dimers` |

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
`families_corrected` and `families_fixedq` rows must be re-scored; the other five sets stand.

## 3. What is left to do

1. Finish the `so3lr-tauri-l` re-score (running).
2. Re-score `so3lr-tauri` (`so3lr-tauri-m-2606`), alone on the card.
3. Re-score `so3lr` v1 — at minimum `families_corrected` and `families_fixedq`; re-scoring all seven costs the same
   twenty minutes and keeps one provenance stamp per row, which is cheaper to defend than a split one.
4. **Gate, before `lrbench results add`:** `n_failed == 0` on all seven sets for the three SO3LR-Tauri rows; for
   `so3lr` v1, `n_failed` on the two re-scored sets must fall to its element coverage and the set's first-error note
   must not be a `JaxRuntimeError`. `aimnet2` and `oracle-refs` keep their counts by design.
5. Then `lrbench results add` for all sixteen, `lrbench results render`, and rebuild the explorer from the committed
   rows.

Keep every re-score on `bb3519d` so the rows stay comparable with the thirteen that are already good.

## 4. The two environment faults, and how to recognise them

Both were invisible in the exit status. Both are ours, not the models'.

- **A missing `LRBENCH_DPA4_DIR`** in the first attempt. The run completed and scored.
- **GPU memory contention.** The card is a 16,380 MiB RTX 4060 Ti that also carries the desktop compositor and an
  editor's GPU process, about 900 MiB together. JAX preallocates most of the remainder, so a second process holding
  memory makes its CUBIN load or command-buffer instantiation fail mid-run, structure by structure, while the
  process still exits 0.

The `XLA_PYTHON_CLIENT_MEM_FRACTION=.92` used for the re-score is **not** by itself the cause: `so3lr-tauri-s`
re-scored clean under it. The contaminated `so3lr-tauri` re-score began 92 s after the previous model's process
exited, which is the likelier fault — the card had not been released. The rule that follows is about scheduling, not
about the fraction: **one scoring process on the card at a time, with a gap for the previous one to release memory,
and nothing else compute-heavy alongside it** (no headless browser checks, no priors jobs).

Reading the logs:

- Two `cuda_executor.cc:1182` *"Failed to allocate device memory of 14.33GiB … 12.89GiB"* lines at startup are
  JAX's preallocation ladder stepping down to a size that fits. They are **benign** and appear in clean runs too —
  the clean `so3lr-tauri-s` re-score has exactly these two and nothing else.
- What marks a contaminated run is `JaxRuntimeError` inside the result's `notes`, or `Autotuning failed for HLO`
  repeated through the log. Count those, not the startup lines.
- `unsupported element`, `species not implemented` and `outside the oracle's domain` are model and set properties.
  Keep them, flag them, never re-score them away.

## 5. Provenance note

The first `so3lr-tauri` v3 result was overwritten by its own re-score at 21:46, so its original failure count cannot
be re-read; it was reported in-session as 362 and is unverifiable. The two surviving v3 results were copied to
`runs-v3/backup-v3/` before the running batch could overwrite them (`so3lr-tauri-s`, 12 failures; `so3lr-tauri-l`,
1427). An in-session summary table also mapped the three Tauri variants to the wrong counts — by `created` stamp and
per-set breakdown, `so3lr-tauri-l` was the 1427 and `so3lr-tauri-s` the 12, not the reverse. Trust the JSONs.
