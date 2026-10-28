# Round 9 code review: dataop-logger

## Verdict

The suite passes: **59 passed in 5.11s**.

There are **no high-severity issues** and I found no new high- or medium-severity regression introduced by this commit. The atomic batch/revision fix is sound. The main Round 8 medium finding is only partly fixed: idle plateaus are now separate suggestions, but adjacent suggestions still make **Cut all** exclude the movement boundary along with every other frame.

## Round 8 audit

| Round 8 finding | Status | Evidence |
| --- | --- | --- |
| M1: idle spans bridge movement | **partly fixed** | `review.py:425-443` splits runs of quiet windows, and the new boundary tests cover jumps. However, `[0]*15 + [10]*15` still returns `[(0, 15), (15, 30)]`; accepting both suggestions covers 30/30 frames. The proposals touch, so their training-mask union still removes the transition that the split was meant to preserve. Leave at least the action/observation boundary frame(s) outside both spans and assert the union produced by Cut all, not just the tuple list. |
| M2: finder revision check and batch were non-atomic | **fixed** | `review.py:216-240` validates every item before one locked `transaction(expect_rev)`, and `review.py:446-467` passes the viewer revision into that batch. `viewer.py:179-193` reports the single batch result. An invalid second item wrote no file; two concurrent `expect_rev=0` writers produced one complete two-edit batch and one `ReviewConflict` (`rev=1`, two edits). |
| L1: legacy non-cut proposals invisible/unresolvable | **fixed** | `viewer.js:38-40,286-290` renders non-cut proposals with Accept/Dismiss controls. `validate.py:371-372,394-395` reports cut suggestions and other proposals separately. |
| L2: draft switching could lose edits or resurrect drafts | **partly fixed** | Dirty drafts now survive run switching (`viewer.js:231-239`), opening another item is blocked (`viewer.js:311-327`), and Cancel/Done/Escape clear the saved draft (`viewer.js:330-340,376,440-455`). But Cut clears `sel` and the draft before awaiting the write (`viewer.js:335`), with no restore path like `saveLabel()` has at `viewer.js:164-175`; a 409/network failure loses the precisely selected span while the 409 message says the draft was kept. Retain and restore the selection when Cut fails. |
| L3: overlapping cuts double-counted | **fixed** | The viewer counts a frame union with a bitmap (`viewer.js:33-36,283-285`), and inspect does the same (`validate.py:374-377`). |
| L4: stored interpretation history/sequence metadata under-validated | **partly fixed** | `review.py:315-346` now checks history-list shape, interpretation fields, unique/ranged `accepted_seq`, and `seq`; the prior crash and ambiguous sequence cases are covered. It still accepts falsey malformed current interpretations because `e.get("interpretation")` skips `[]`, `{}`, and `""`, and string-only checks allow empty/over-limit history fields that `interpret()` cannot write (`review.py:336-342`). Validate field presence explicitly and reuse the full interpretation constraints for current and historical entries. |
| L5: fractional click-selected frame ranges | **fixed** | `viewer.js:407-413` rounds FPS to an integer width and keeps both boundaries integral. |
| L6: duplicate suggestions reported as newly created | **partly fixed** | Explicit statuses now flow through `review.py:202-240`, the CLI (`cli.py:113-119`), and viewer summary (`viewer.js:145-151`). Status precedence is still wrong: `_prior_suggestion()` checks the oldest matching agent edit before active human cuts (`review.py:205-212`). I reproduced Keep on an agent suggestion, followed by a manual cut of the same span; a later suggestion reports `previously_kept` instead of `already_cut`. Check active cuts first, then proposal history. |

## New findings by severity

### High

None.

### Medium

None introduced by these fixes. The remaining medium issue is the partly fixed M1 above.
