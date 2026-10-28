# Round 16 functional test — Codex

Tested commit: `0a254223135c7079be473532530d7829ead8294c`  
Dataset: `/tmp/dataop_r16`  
Interpreter: `/Users/chadster/lerobot/.venv/bin/python` with `PYTHONPATH=src`

## Verdict

**FAIL — not ready for release.** The recorded dataset remains valid and the 71-test suite passes, but two normal workflows violate data-integrity expectations:

1. `remap` silently loses the dataset's list schema (entries, descriptions, and aliases) while reporting success and producing a dataset that validates `OK`.
2. A multi-field `annotate` command is not atomic: if a later requested field is invalid, the command exits with an error after saving earlier fields.

There are also two medium-severity gaps: CLI list management omits `describe` and `delete`, and idle detection assumes degree-like joint positions even though LeRobot deployments may use normalized, radian, Cartesian, velocity, or custom state/action spaces.

The supplied fixture contains 20 LeRobot episodes and 20 committed dataop runs, not the 30 stated in the test request. I tested all 20 on-disk runs and did not count that fixture discrepancy as a product failure.

The repository was clean at `0a25422` when testing began. During the run, unrelated modifications and new batch-related files appeared in the working tree. I did not create, modify, or revert them. All tests after that point ran from a clean `git archive 0a25422` at `/tmp/dataop_logger_r16_0a25422`; the viewer process had already imported the clean commit before the concurrent changes appeared.

## High severity

### H1. `remap` silently drops `dataop/lists.json` schema data

Exact commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli remap /tmp/dataop_r16 /tmp/dataop_r16_remap \
  --repo-id org/renumbered --revision r16

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli validate /tmp/dataop_r16_remap --json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli lists /tmp/dataop_r16

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli lists /tmp/dataop_r16_remap
```

What happened:

- All 20 runs reported a unique episode mapping.
- Destination validation reported `"ok": true`, 20 runs, 20 episodes, and no warnings.
- The source list contained:
  - failure type `r16_contact`, alias `r16_collision`, description `contact with obstacle`;
  - task segment `r16_approach`, description `move toward target`.
- Both entries, their descriptions, and the alias were absent in the remapped dataset. The destination reconstructed only names currently used by runs.

What should happen:

`remap` must carry the dataset-level list schema forward (including unused agreed entries, descriptions, aliases, and revision semantics), or explicitly reconcile it with an existing destination schema. It must not report a fully successful remap while silently discarding schema data.

### H2. Failed multi-field `annotate` persists a partial edit

The selected run has only a policy drive span, so `--takeover-at` is invalid.

Exact command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli annotate /tmp/dataop_r16 \
  4302000a-653e-4df1-ac3c-f738c59c1468 \
  --segments '[{"name":"r16_approach","start_s":0,"end_s":1},{"name":"grasp","start_s":1,"end_s":13.567}]' \
  --takeover-at 2 --by codex-r16-partial
```

The command exited 2 with:

```text
error: this run has no switch from policy to another drive mode
```

Follow-up command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli runs /tmp/dataop_r16 \
  --run 4302000a-653e-4df1-ac3c-f738c59c1468
```

What happened:

Despite the failed command, `runs` showed that the new `r16_approach`/`grasp` segment plan had been saved with `segments_set_by: "codex-r16-partial"`. The invalid takeover was rejected only after the segment edit committed.

What should happen:

Validate every requested annotation before writing anything, then commit all requested fields atomically (or roll back earlier writes when any later field fails). A nonzero command must not leave a hidden partial result.

## Medium severity

### M1. CLI list management does not expose `describe` or `delete`

Exact commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli lists /tmp/dataop_r16 describe failure_types r16_contact \
  --description 'updated description'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python \
  -m dataop_logger.cli lists /tmp/dataop_r16 delete task_segments r16_approach
```

What happened:

Both exited 2 during argument parsing:

```text
argument action: invalid choice: 'describe' (choose from show, add, rename)
argument action: invalid choice: 'delete' (choose from show, add, rename)
```

What should happen:

The agent-facing CLI should expose the same list lifecycle as the viewer API: add, rename, describe, and delete. Deleting an unused entry should succeed; deleting an in-use entry should return the existing clear error (`'grasp' is used by 20 runs; change those first`). The underlying list functions and viewer endpoints already implement both operations.

### M2. Idle detection is not unit- or state-space-general

Exact command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/chadster/lerobot/.venv/bin/python -c \
  'import numpy as np; from dataop_logger.review import idle_spans; motion=np.linspace(0.0,1.0,60,dtype=np.float32)[:,None]; print(idle_spans(motion, rate=30))'
```

What happened:

```text
[(0, 60)]
```

A state moving continuously through its entire normalized range for two seconds was classified as entirely idle. The implementation assumes `observation.state` means measured joint position, applies a fixed `1.5` threshold in unknown units, and describes results as “arm still” / “measured joints.” The fallback applies the same threshold to `sent_action`, regardless of action space or units.

What should happen:

Idle detection must derive meaningful scaling from the declared schema/bounds, accept a deployment-specific threshold/model, or refuse to make a built-in suggestion when the state/action semantics are unknown. It must not interpret arbitrary LeRobot state vectors as degree-valued arm joints.

## Low severity

### L1. No-op `find_idle` writes a new revision

After all detected idle spans were already suggested, cut, or kept, this exact viewer request was sent:

```bash
curl -sS -X POST http://127.0.0.1:8771/dataop/api/review \
  -H 'Content-Type: application/json' -H 'X-Dataop: 1' \
  --data '{"run_id":"04ec0a92-d125-4989-a6dc-941481b73095","rev":12,"op":"find_idle","what":"find idle frames"}'
```

What happened:

The response contained `created: 0`, no undo step, and no changed edit, but advanced the review from rev 12 to rev 13.

What should happen:

A no-op should preserve the revision. Advancing it needlessly causes otherwise valid concurrent edits to receive a stale-revision conflict.

## Passed coverage

| Area | Result |
|---|---|
| `validate` | Initial and post-review source validation: `OK: 20 runs, 20 LeRobot episodes`; remapped dataset also validated with no errors or warnings. |
| `inspect` and direct data check | Matched 14 success / 6 failure, 6 takeovers, 11,614 frames, 9,650 base frames (5.36 min), and 1,964 expert frames (1.09 min). Review counts also matched one label, 10 cut frames, and six open suggestions. |
| `runs` / `annotate` | A valid annotation round-tripped exact segment frames (0/300/600/1168), failure frame 600/type `no_grasp`, and takeover frame 645. |
| `events` / `note` | Drive, segment, failure, and end events used the effective edited frames; an agent note replaced the recorded failure note with correct attribution. |
| `labels` / `interpret` | Uninterpreted context was correct; first interpretation saved; silent replacement was refused; `--replace` preserved the old reading in `interpretation_history`. |
| Lists that are exposed | CLI add/rename preserved description and alias. Viewer `lists_add`, `lists_describe`, `lists_rename`, unused delete, in-use delete rejection, and revision updates behaved correctly. |
| `find-idle` / `suggest-cut` | Real dataset finder created eight proposals and repeated detection was idempotent at the edit level; manual cut suggestion stayed proposed. |
| Viewer load and POST API | Loaded all 20 runs. Exercised `add`, `note`, `decide`, `decide_many`, `find_idle`, `replace`, `undo`, and all `lists_*` operations with the same bodies/header as `viewer.js`. Undo restored the original label id, span, text, driver, and segment. |
| Viewer conflicts and bad input | Stale run rev returned 409 with current state; invalid zero-length label returned 400; missing `X-Dataop` returned 403. |
| Video ranges | `bytes=0-99` and `bytes=-64` returned correct 206 ranges/lengths; an out-of-range request returned 416 with `Content-Range: bytes */117215086`. |
| `remap` binding | All 20 fingerprints mapped uniquely and run-level manifests/reviews copied; destination manifest binding and recorded data validated. List-schema preservation failed as H1. |
| Pytest | `71 passed in 5.16s` using `python -m pytest -q -p no:cacheprovider`. |
| Generality checks | Camera keys, video files/offsets, task text, segment/failure names, and FPS were derived from metadata/manifests/lists. The unit-dependent idle heuristic failed as M2. |

The viewer server on port 8771 was stopped after the HTTP tests.
