# Round 11 decisions

All findings are fixed. 63 tests pass, and the browser regression run is clean (undo after reload, live defaults on resize).

| Finding | Decision |
| --- | --- |
| M3 partly fixed: snapshots only shallow-checked | History snapshots are validated as full edits (`_valid_edit`): id, kind, status, source, canonical payload, `accepted_seq`, readings. Step ids must be unique, `at` must be text and `t` a positive int. |
| L3 partly fixed: one oversized step kept | A step whose JSON exceeds the cap is not kept for undo, and the oldest steps are dropped until the history fits. Validation enforces the byte cap too. |
| New M1: a stale step blocked Cmd+Z | `undoable()` offers a step only while every edit it touched still equals its `after` snapshot. A later change hides it, and older independent steps stay reachable (tested with a CLI interpretation after a viewer step). |
| New M2: defaults not refreshed on resize | Driver and segment defaults are recomputed when a new label's span is resized, unless the person already chose them. A saved label keeps its values. |
