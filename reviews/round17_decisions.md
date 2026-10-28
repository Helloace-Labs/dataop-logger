# Round 17 decisions (design review of batches, import and export)

| Finding | Decision |
| --- | --- |
| H1 import could not be stored honestly | Added `unknown` as a control mode and an outcome, and `from_dataset_timestamp` as a timestamp kind. The logger still records only the live values. A known base action no longer needs an inference record (`inference_id = -1`). Success comes from `any(next.success)`. HIL mapping: non-intervention `action` is the base action, intervention `action` the expert action. `next.done` is not mapped. |
| H2 the watcher preset did not match the files | The raw `.npy` maps with `covers: policy_frames`: its row count must equal the policy-driven frames, and rows land on those frames in order. Segments and failure kinematics stay out of the preset. |
| H3 export time series | Accepted for the export build: every run is cut into maximal contiguous kept spans after all selectors, each span becomes its own episode written through LeRobot (`create`/`add_frame`/`save_episode`/`finalize`), transition features (`next.*`) are dropped, and `--val` writes two datasets. Several roots must share fps, robot type and features. |
| H4 residual recipe | Accepted: it is called "full-action supervised residual", needs recorded base actions, and `--zero-on-success` uses only policy frames with `sent == base` and no safety change. The SO-101 correction build stays a user script; the hook can now return a keep mask, actions and extra features. |
| H5 determinism | Accepted for export: one effective-run snapshot (with reviewed drive modes) feeds the viewer, the training mask and export. The export record holds digests of lists, manifests, ticks, review state, source LeRobot files and the output, plus the hook's file digest and result digest. |
| M1 one batch registry | Done: a single `dataop/batches.json` with the run-to-batch map under one lock. Edits to locked runs are refused in `review.transaction`. Locking needs every run reviewed and no open suggestions, or `--force` (recorded). |
| M2 copy with `delete_episodes` | Dropped: `--into` copies the folder byte for byte. No merging during import. |
| M3 DAgger target rule | Accepted: `expert_action` where valid, otherwise the frame is dropped with a reason. |
| Low: smaller first release | Followed in order: batches, import (plain, lerobot-hil, watcher), then export (filtered, dagger, residual). Push/pull to the Hub is deferred. The review lock stays, since the owner wants export to follow a finished review. |
