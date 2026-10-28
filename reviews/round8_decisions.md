# Round 8 decisions

All findings were accepted and fixed. 59 tests pass. Browser checks cover: idle suggestions and their summary notice, cut all, label drafts (new and existing) surviving a run switch, Done discarding a draft, empty-text rejection, a two-window delete conflict, and a cross-origin iframe being blocked (Chrome shows "127.0.0.1 is blocked").

| Finding | Decision |
| --- | --- |
| M1 idle spans bridged a movement | `idle_spans` now turns each run of consecutive quiet one-second windows into one stretch. A movement breaks the run, so plateaus on either side stay separate and never merge. Table-driven tests cover a jump between plateaus, a one-frame spike, shorter than one window, exactly one window, always moving, and a short second plateau. |
| M2 find_idle revision check not atomic | Candidates are computed first, then `add_suggestions` validates, deduplicates and appends the whole batch in one locked transaction with `expect_rev`, bumping the revision once. A stale batch raises 409 and writes nothing (tested). |
| L1 legacy proposals invisible | Pending proposals of other kinds show under "Other corrections" with Accept and Dismiss. `inspect` reports `open_cut_suggestions` and `other_open_proposals` separately. |
| L2 draft lifecycle | Any edited selection (new or existing label) is marked dirty and kept per run when switching runs. Done, E, Cancel, Escape and a successful save discard it. Opening another label or cut while a draft is dirty asks the person to save or cancel first. |
| L3 overlapping cuts double-counted | The page and `inspect` count the union of cut frames. |
| L4 history and sequence not validated | Interpretation history must be a list of valid readings (train, meaning, by, at). `reason` and `by` must be text. `accepted_seq` values must be unique integers in `[1, seq]`. |
| L5 fractional click span | The click width is `round(fps)` frames, clamped to the run. |
| L6 duplicate suggestions looked new | Suggestion calls return a status: `created`, `already_suggested`, `already_cut` or `previously_kept`. The CLI prints it, and the button shows a summary such as "Found 9 idle stretches: 0 new, 9 already cut." A span a person kept is not suggested again by any agent. Keep's tooltip says so. |
| Product: card mixed cuts and other corrections | Split into "Cut · N s left out of training" and "Other corrections". |
