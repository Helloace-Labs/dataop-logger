# Round 20 code review and functional test

## Verdict: request changes

Commit `218e239` passes the existing suite and the exercised happy path, but cross-dataset selection can silently export runs the user did not choose. Four medium issues also affect stale reporting, Watcher import, blend data, and the CLI.

The repository advanced to a later commit while testing, so final verification was pinned to an exact `git archive 218e239` at `/tmp/r20_commit_218e239`. The reviewed diff was `git diff ab5a00d..218e239`.

## High

### H1 — Cross-dataset picks lose the dataset identity and include unselected runs

`viewer.py:400-418` groups picks by root, then flattens them back to one global set of run IDs. `export.py:347` applies that set independently to every source. Dataset copies/remaps legitimately share run IDs, so choosing run X from A and run Y from B exports both X and Y from both datasets. A sidecar-preserving demo copy reproduced it:

```bash
curl -sS -H 'Content-Type: application/json' -H 'X-Dataop: 1' \
  -H 'X-Dataop-Root: %2Fprivate%2Ftmp%2Fdataop_r20_demos' \
  --data '{"op":"train_preview","recipe":"teleop","into":"/tmp/r20_collision_preview","runs":[{"root":"/private/tmp/dataop_r20_demos","run_id":"046b8644-36c2-44e3-a5b2-52bb6b3ba9a2"},{"root":"/private/tmp/r20_demos_clone","run_id":"4c733b7b-f6fb-4216-8651-1f0b15342730"}]}' \
  http://127.0.0.1:8773/dataop/api/review
# chosen: 2; preview: "runs":4, "runs_used":4, "episodes":4, "frames":1260
```

Fix: preserve `(resolved_root, run_id)` throughout planning, filtering, provenance, and snapshot lookup. Pass `runs_by_root` (or fully qualified picks) to `export()` and filter each batch only with that root's IDs. Also change the writer's snapshot lookup from run ID alone to `(root, run_id)`. Add a two-copy collision test asserting exactly two runs are planned and written.

## Medium

### M1 — A combined set is out of date only when viewed from the changed dataset

`export.training_sets()` computes review digests only for its `root` and filters the export record to `mine` (`export.py:492-514`). After changing one exported DAgger run, the same set gave:

```text
train_list on dataop_r20_dagger -> changed_runs=1, out_of_date=true
train_list on dataop_r20_demos  -> changed_runs=0, out_of_date=false
```

Fix: read every run in `dataop_export.json`, resolve each recorded source root, and compare every recorded review digest. Return one global `changed_runs/out_of_date` result from every source registry, as round 19 already requires for lock staleness. Add a two-root test that edits one root and queries both.

### M2 — The Import dialog/API cannot supply the mapping required by this Watcher dataset

The CLI mapping produces the correct dry run:

```bash
DATAOP_HOME=/tmp/r20_home PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli import /tmp/dataop_r20_dagger --preset watcher \
  --map /tmp/r20_watcher_map.json --dry-run
# 40 runs; 31 success, 9 failure; 19,869 policy frames; file-mtime time mapping present
```

But `viewer.py:348-375` ignores a `mapping` member and forwards only the preset. `auto` detects this source as `plain`; forcing `watcher` still uses nonexistent `{root}_raw`/`{root}_results_*` paths. Both API previews reported 40 `unknown` outcomes and zero policy frames, even when the request included the absolute mapping.

Fix: expose an advanced mapping override in the dialog, validate it as an object, and pass it to both preview and background import. Detection/preview must use the same effective mapping that the real import will use. Add an API test with non-adjacent absolute Watcher side files, including `time`.

### M3 — LoRA drops blend-only DAgger data

`export.py:95-97` calls a run teleop whenever it has no pure `base` frame. A `blend` frame is shared policy/expert control and is already treated as a takeover elsewhere. The same two-frame snapshot is kept by `dagger` but dropped by `lora`:

```bash
PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python - <<'PY'
import numpy as np
from dataop_logger.export import select
s={'n':2,'effective':{'rejected':False,'termination_reason':'success'},'train_mask':np.ones(2,bool),'sent':np.ones((2,1),np.float32),'expert':np.full((2,1),2,np.float32),'mode':np.array(['blend','blend'],object),'expert_valid':np.ones(2,bool),'segments':[]}
for recipe in ('dagger','lora'):
    print(recipe, int(select(s,recipe,{}, {'task_segments':[]})[0].sum()))
PY
# dagger 2
# lora 0
```

Fix: treat both `base` and `blend` as policy participation when classifying a run for LoRA/teleop, and add blend-only plus mixed base/blend/expert recipe tests.

### M4 — `lora` is implemented but rejected by the export CLI

```bash
PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python -m dataop_logger.cli export \
  --batch '/tmp/dataop_r20_dagger:round20 watcher' --recipe lora \
  --into /tmp/r20_cli_lora --allow-unlocked --dry-run
# exit 2: invalid choice: 'lora' (choose from teleop, filtered, dagger, residual)
```

Fix: add `lora` to `cli.py:133` (prefer deriving parser choices from `export.RECIPES`) and update option help/tests.

## Functional verification

- Viewer API imports: with a test-only adjacent symlink bridge for the unsupported Watcher mapping, Watcher produced `31 success / 9 failure`, `19,869` policy frames, and all 40 `ended_at` values exactly equaled their `.npy` mtimes; demos produced `20 success`, all `11,610` frames expert-driven.
- Cross-dataset previews: teleop `300`, DAgger `274`, LoRA `574` frames; `intervention` was automatically dropped.
- Real `/tmp/r20_lora_real`: `LeRobotDataset` loaded 2 episodes with boundaries `274 + 300`; action max error was `0.0` against expert/demo sources; six frames from both cameras matched their source frames with mean absolute pixel error `0.0062–0.0088` after re-encoding.
- One running import/training job rejected a concurrent start. Unknown workspace roots, existing output folders, and missing `X-Dataop` were rejected without writes.
- Exact-commit headless Chrome with throwaway profile: desktop `1440x1000` and mobile `400x900`; header controls and both dialogs rendered, no page-level horizontal overflow, no JS exceptions.
- Exact-commit pytest: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python -m pytest -q -p no:cacheprovider` — **94 passed in 5.91s**.
