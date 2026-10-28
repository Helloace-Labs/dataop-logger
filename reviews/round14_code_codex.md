# Round 14 verification

## Verdict

The outcome, zero-value data checks, accessibility, pre-run events, and Robot/Operator fixes are correct. Three round 13 findings are only partly fixed: intervention transitions still misstate some mode changes, merging a takeover request can discard its attribution/timing, and mixed equal/non-equal timestamp rows still bias latency percentiles. There are no new high-severity issues. Tests: `64 passed in 5.00s`.

## Audit

| Round 13 finding | Status | Evidence |
| --- | --- | --- |
| M1: failed run shown as “Rescued” | **fixed** | `viewer.js:54-59,404-412` always shows the effective termination reason and makes intervention use a separate badge/drive-mode fact. |
| M2: takeover spans caused false interventions/transitions | **partly fixed** | `viewer.js:347-355` correctly coalesces contiguous expert/blend spans, and `viewer.js:377-382` handles frame-zero and early windows. But `viewer.js:352,380,383` loses the intervention's initial mode and assumes every end returns to policy; see Medium 1. |
| M3: takeover details and unfinished steps disappeared | **partly fixed** | `viewer.js:366-369,386-387` restores unfinished and unmatched events. A matched takeover request keeps only reason/text and can disappear completely when it contains only source; see Medium 2. |
| M4: Data check hid zero/equal values | **partly fixed** | `viewer.js:414-430` always renders latency, safety, and cut rows. `viewer.py:93-95` distinguishes an all-equal run, but still includes equal placeholder rows in a mixed run; see Medium 3. |
| L1: nested/incomplete sliders | **fixed** | `viewer.js:310,322-324,481-484` leaves one focusable valued slider and exposes selection frames/instructions; `viewer.html:47` adds visible focus. |
| L2: pre-recording events lost meaning/order | **fixed** | `viewer.js:361-363,386-400` preserves frame `-1`, renders “before,” sorts it first, and uses an explicit stable tie order. |
| L3: Robot and Operator were ambiguous | **fixed** | `viewer.js:421-427` renders separate optional rows. |

## High

None.

## Medium

### 1. Merged interventions still produce false start and handback events

**Evidence:** `src/dataop_logger/viewer.js:347-355,377-383`

For `base -> blend -> expert -> none`, `interventions()` correctly creates one span, but `full` becomes true because expert appears later. The event at the original blend boundary therefore says “Rescue takes over,” and the end says “Policy back in control” even though the next mode is `none`. The same false handback occurs for any intervention followed by `none`; a pre-takeover warning after `none` also incorrectly says the policy went wrong.

**Fix:** Retain the initial mode and internal mode transitions in each intervention, and inspect the segment immediately before/after it. Describe the start from its actual mode, emit “Policy back” only when the next mode is `base`, and use neutral warning text unless policy was actually in control before the intervention.

### 2. Matching a takeover request can silently discard who requested it and when

**Evidence:** `src/dataop_logger/viewer.js:371-382`

A recorded takeover event within one second of an intervention is marked `used`, but the generated event copies only `reason` and `text`. Its `source` and original frame are lost. A request containing only `source="operator:alice"` disappears completely; a request one second before the switch is shown at the switch time without its requester.

**Fix:** Carry the request source and original time into the merged event/note (for example, “requested by operator:alice at 0:03.2”). Do not mark an event consumed unless all meaningful fields remain visible.

### 3. Mixed synthetic/equal timestamps still understate latency

**Evidence:** `src/dataop_logger/viewer.py:77,93-95`

`measured` is run-wide: one positive delta makes it true, while every equal row remains in the percentile input. A run with 90 equal placeholder rows and 10 measured 20 ms rows reports a 0 ms median and can substantially depress p95, despite most usable measurements being 20 ms.

**Fix:** Add per-row timing validity/source to the recording contract and compute percentiles only from measured rows, reporting measured/total counts. Without that metadata, label the statistic as including zero deltas rather than inferring that the entire run is measured.

## Low

None newly introduced.
