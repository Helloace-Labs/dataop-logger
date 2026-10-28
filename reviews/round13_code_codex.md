# Round 13 code review

## Verdict

The round 12 interaction bugs are fixed, except that the accessibility fix is incomplete. The new cards are generic rather than arm-specific, escape displayed run/review data correctly, and labels open correctly from Major events. The main remaining problems are semantic: continuous blend/expert intervention is presented as multiple takeovers with false handbacks, recorded takeover reasons and unfinished steps disappear, and a failed run that ever had expert involvement is called “Rescued.” There are no high-severity findings.

Test result: `64 passed in 5.20s`.

## Round 12 audit

| Finding | Status | Evidence |
| --- | --- | --- |
| 1. One-frame resize expanded to one second | **fixed** | `viewer.js:517-543` records actual movement, limits the one-second fallback to an unmoved new selection, and preserves a one-frame resized/crossed span. |
| 2. New drag appended duplicate selection nodes | **fixed** | `viewer.js:510,518-520` makes `box` mutable and assigns the first created element back to it. |
| 3. Near-end default did not start at playhead | **fixed** | `viewer.js:247-250` anchors `lo` at `frameOf(r)` and truncates the two-second span at run end. |
| 4. Selection covered labels and cuts | **fixed** | `viewer.html:117-118,123-133,170-172` puts label/cut targets above the selection and keeps handles below the label lane. |
| 5. Pointer-only controls and small targets | **partly fixed** | `viewer.html:127-133` adds touch suppression and 16 px handles; `viewer.js:324,455-456,588-592` exposes the playhead value and adds `[`/`]` edge controls. But the ARIA slider itself is not focusable, while the focusable parent is a second slider without value attributes, and the selection exposes neither boundary value. See Low 1. |

## High

None.

## Medium

### 1. A failed or timed-out run with any expert span is reported as “Rescued”

**Evidence:** `src/dataop_logger/viewer.js:54-59,381-389`

Both the outcome pill and Run card use only `r.takeovers.length` to substitute “rescued” for the recorded termination reason. A run can enter blend/expert control and still end in failure, timeout, operator abort, or system error; it will be presented as rescued and the actual result is omitted from the Run card. Blend-only assistance is also called rescued.

**Fix:** Always derive Result from the effective termination reason/task success. Report expert involvement separately as “rescue used” or “shared control used”; use “rescued” only if the data model has an explicit successful-rescue outcome.

### 2. Takeover spans are mistaken for independent intervention episodes and create false transitions/windows

**Evidence:** `src/dataop_logger/viewer.js:349-360,387-389`

Manifest takeovers are split whenever control mode or expert source changes. For adjacent blend `[5, 7)` and expert `[7, 10)`, the page reports two takeovers, adds a second “policy went wrong” window at frame 7, and emits “Policy back in control” and “Rescue takes over” at the same frame even though expert involvement was continuous. A run beginning in expert mode also gets both “Rescue in control” and “Rescue takes over” at frame 0. Conversely, a takeover in the first second gets no available pre-takeover highlight because line 356 requires `start_frame > round(fps)`.

**Fix:** Build UI intervention episodes from the full mode segments, coalescing contiguous expert/blend spans as appropriate. Emit a handback only when the following mode is actually base, treat frame-zero expert/blend as the initial state rather than a transition, and show the partial `[0, start)` warning window for every nonzero early takeover.

### 3. Major events silently drops recorded takeover detail and unfinished steps

**Evidence:** `src/dataop_logger/viewer.py:85-87`; `src/dataop_logger/viewer.js:351-364`

Every recorded `takeover` event is filtered out, so its original frame, reason, source, and text disappear; a derived control-mode transition is not equivalent. For example, an operator request one frame before control changes with reason “object dropped” is no longer visible. The replacement for the Steps card also emits only completed/failed subgoals, so a step left `unfinished` at run end vanishes entirely.

**Fix:** Merge recorded takeover events with nearby derived transitions while preserving their reason/text, or show both when they do not describe the same moment. Add an “unfinished at run end” major event for unfinished subgoals. Test a request preceding mode change, a takeover at frame 0, and a run ending with an open step.

### 4. Data check suppresses valid zero values and cannot distinguish unavailable timing from a clean run

**Evidence:** `src/dataop_logger/viewer.py:77,93-94`; `src/dataop_logger/viewer.js:391-405`

An all-equal `obs_ns`/`sent_ns` run gets `latency_ms = null`; zero safety-filter frames and zero cut time are also filtered out. Thus a common clean or synthetic-timestamp run has no Data check section at all, and the team cannot tell “zero,” “not measured,” and “not checked” apart. If equal/synthetic rows are mixed with measured rows, their zero deltas are included in the percentiles and can understate p50/p95.

**Fix:** Return explicit timing validity/source plus p50/p95, and show all three checks even when they are zero. If legacy synthetic timestamps are supported, exclude only rows explicitly marked synthetic; otherwise report an all-equal measured run as `p50 0.0 ms · p95 0.0 ms`. Label the percentiles as p50/p95 rather than “slowest 5%.”

## Low

### 1. The ARIA model still has two incomplete nested sliders

**Evidence:** `src/dataop_logger/viewer.js:310,324,455-456`; `src/dataop_logger/viewer.html:112,127-133`

The parent timeline is the only focusable element and has `role="slider"` but no min/max/current value; the child ruler has those values but no `tabindex`. There is also no visible focus rule for the timeline and no accessible start/end value for the selection. Keyboard shortcuts exist, but a keyboard or screen-reader user is not told enough to discover or verify them.

**Fix:** Make one playhead element the single focusable slider with all value attributes and visible focus, and expose the selected start/end plus `[`/`]` instructions through labelled controls or an ARIA description.

### 2. Pre-recording events lose their meaning and tie order

**Evidence:** `src/dataop_logger/viewer.py:85-87`; `src/dataop_logger/viewer.js:348,363-372`

An event recorded at frame `-1` is clamped to frame 0 and displayed as `0:00.0`. Because `collect()` drops event id/time and the final sort uses only the clamped frame, it can appear after synthetic frame-zero state/step/takeover entries even though it happened before recording.

**Fix:** Preserve the pre-run sentinel and event sequence/time, render it as “before recording,” and use an explicit tie-break key rather than category insertion order.

### 3. Robot and operator are combined under one ambiguous label

**Evidence:** `src/dataop_logger/viewer.js:397-402`

The `Robot` row renders `robot · operator` with no role labels; if only an operator exists, their pseudonym is shown as the robot. A deployment team unfamiliar with the manifest cannot tell which value is which.

**Fix:** Render separate Robot and Operator rows (or label each inline). Keep both optional.
