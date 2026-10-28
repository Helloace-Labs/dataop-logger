# Round 21 code re-check

## Verdict: fixes verified

At `571bacb`, all five Round 20 behaviors are fixed. No new functional regression was found in `git diff 218e239..571bacb`. The full suite passes: **98 passed in 6.14s** (`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python -m pytest -q -p no:cacheprovider`).

- **H1 fixed.** Repeating the two-copy viewer preview on port 8773 returned exactly `runs=2`, `runs_used=2`, `episodes=2`, `frames=630` (formerly 4/4/4/1260). A real same-run-id residual export from two roots, with the second root's base actions shifted by 100, wrote 12 frames with distinct per-root base actions (`6..13` versus `106..113`) and residuals (`1` versus `-99`), confirming the writer's `(root, run_id)` lookup. The added test covers per-root planning only: it uses `dry_run=True`, selects different run IDs, and does **not** cover the writer lookup.
- **M1 fixed.** After the existing DAgger source edit, `train_list` from both `/tmp/dataop_r20_dagger` and `/tmp/dataop_r20_demos` returned `changed_runs=1`, `out_of_date=true`, `runs=2`. The added two-root real-export test covers this bug.
- **M2 fixed.** With `/tmp/r21_watcher_source` separating the dataset from its side files, the mapped API preview returned 40 episodes, 31 success / 9 failure, 19,869 policy frames, the absolute file-mtime mapping, and no missing files; the unmapped preview returned 40 unknown outcomes and zero policy frames. A non-object mapping was rejected. The CLI dry run matched. **No automated test was added for API mapping/preview/start or missing-side-file reporting.**
- **M3 fixed.** The original blend-only repro now gives `dagger=2`, `lora=2`; mixed `base/blend`, `base/expert`, and `blend/expert` cases also give `lora=2`. The new test covers blend-only and expert-only classification, but not the requested mixed-mode cases.
- **M4 fixed.** The original CLI command accepts `--recipe lora` and exits 0 (9 episodes, 3,118 frames on the Round 20 data). The added CLI test covers the parser regression.

## New finding

- **Low — incomplete regression coverage.** H1's writer branch and all of M2 lack automated coverage, and M3 lacks the promised mixed base/blend/expert cases. The implementations passed manual repros, but these branches can regress without failing the suite.
