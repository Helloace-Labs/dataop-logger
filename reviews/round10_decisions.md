# Round 10 decisions

All findings were accepted and fixed. 61 tests pass. A browser run covers undo after a reload, undo across runs, Cut all and the idle finder as single steps, and Cmd+Z inside the text box staying as text undo. This round also includes the owner's design pass on the label panel: one line for the span; a text box; driver and segment chips that default to who drove and the recorded step covering most of the selection; and Save · Cut · Cancel, with explanations moved to tooltips.

| Finding | Decision |
| --- | --- |
| M1 failed undo removed from the stack | The page no longer keeps its own stack. Each run's state includes `undoable` (steps not yet undone) from the server, and Cmd/Ctrl+Z takes the newest across runs. A failed undo leaves the step where it was, and a successful one disappears because the server says so. |
| M2 reload lost undo | Same fix: the list comes from `review.json` history, so it survives reloads. Steps carry `t` in nanoseconds, so steps on different runs order correctly. |
| M3 history not validated | `check_review` validates history: size, ids, `what`, `t`, `undone`, unique touched ids per step, and snapshot shape. `undo()` takes the manifest and runs `check_review` on the result before saving. An undo that would restore an invalid edit raises and writes nothing (tested). |
| L1 several steps in one recording | A recording allows one transaction; a second raises `ReviewError`. |
| L2 cross-run undo hid a draft | Undo switches runs through `select()`, which restores a saved draft, and then stops if that draft has unsaved changes. |
| L3 history size | History is capped at 200 steps and 2 MB of JSON, dropping the oldest first. SPEC notes that undo history keeps old label text until it is pruned. |
