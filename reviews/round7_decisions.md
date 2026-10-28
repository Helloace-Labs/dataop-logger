# Round 7 decisions

The product owner also asked, in the same round, for frame cutting and an AI feature that finds idle frames. Those are built here together with the fixes. 52 tests pass. A Playwright walk covers idle suggestions, cut all, label with a new segment and driver (the draft survives chip clicks), multi-line text with Cmd/Ctrl+Enter to save, resizing a label, empty-text rejection, and a two-window delete conflict.

| Finding | Decision |
| --- | --- |
| Product: text looks required but isn't | Text is now required for every label, in the backend and the page. Driver and segment stay optional. |
| Product: editing clears the agent's reading without warning | The panel says so before saving. |
| Product / M4: driver and segment look authoritative | The panel now says they are notes for the agent and that only the agent's reading changes training. They are context, not corrections, and the reader is unchanged. |
| Product / L3: no resize; global segment list | Dragging while a label is open changes its span, saved through `replace_edit`. Segment chips are this run's steps first, then segments used on the same task. |
| Product / M3: legacy corrections invisible; `propose` creates unresolvable work | `propose` and failure-start proposals are removed. The only agent suggestions are cuts (`find-idle`, `suggest-cut`), and the page resolves them with Cut or Keep. Accepted structured corrections and cuts are listed under "Left out of training" with Undo. The viewer API adds only labels and cuts. |
| M1 decide re-aims an interpreted label | `decide` accepts only proposals; accepted edits can only be undone. Label changes go through `replace_edit`, which drops the interpretation. |
| M2 silent reinterpretation | `interpret` refuses a label that already has a reading unless `replace=True` (`--replace`), and keeps the old one in `interpretation_history`. `--by` names the agent. |
| M5 clickjacking | Every response sends `Content-Security-Policy: frame-ancestors 'none'` and `X-Frame-Options: DENY` as HTTP headers. The meta CSP with a per-load nonce stays for scripts. |
| M6 stale editor after 409 | The page re-renders the run from the returned state. A draft for a label deleted elsewhere becomes a new label, and the message stays until the next action. |
| L1 text optional | Fixed as above. |
| L2 form state and Enter | The whole form lives in the draft and is updated on every input. Enter adds a line; Cmd/Ctrl+Enter saves; Escape cancels. A new label's draft is kept when switching runs. |
| L4 stored review not validated | `check_review` re-validates every stored edit against the manifest (canonical payloads, unique ids, accepted_seq, interpretation shape) inside `effective()`. `validate`, the reader and `train_mask` fail closed, and the viewer shows the error and disables editing for that run. |
| L5 README says read-only | Rewritten: recorded data is never changed, labels and cuts are saved next to the run, and `--read-only` is documented. |
| Owner request: cut frames | A cut is an accepted `exclude` edit: hatched on the run bar, left out through `train_mask`, undoable. Frames are never removed from the recording. |
| Owner request: AI feature, idle frames | **✦ Find idle frames** runs the built-in finder on measured joint positions (`observation.state`; it falls back to commands). A frame is idle when no joint moves more than 1.5 within any one-second window, and a suggestion needs at least 1 s. Suggestions are agent proposals with a short description and `by`. A user's own agent writes the same suggestions with `dataop-logger suggest-cut`. |
