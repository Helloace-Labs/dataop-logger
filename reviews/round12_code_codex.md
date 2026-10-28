# Round 12 code review

## Verdict

The separate scrub strip and relaxed drive-mode validation are implemented correctly, and the core move/resize/crossing calculations clamp safely. However, the new pointer state machine has several user-visible edge cases: a one-frame resize is replaced by an unrelated one-second span, fresh drags can leave a trail of duplicate selection elements, and the default selection is not actually anchored at the playhead near the end of a run. There are no high-severity findings.

Test result: `64 passed in 4.98s`.

## High

None.

## Medium

### 1. Releasing a resize at a one-frame span moves both edges and expands it to one second

**Evidence:** `src/dataop_logger/viewer.js:478-490`

The `hi - lo < 2` click fallback runs for every pointer mode, including `start` and `end`. At 30 fps, resizing `[10, 70)` by dragging its start to frame 69 should produce `[69, 70)`, but pointer-up replaces it with roughly `[54, 84)`. The same happens when an edge crosses the other and is released one frame beyond it. This violates both resize and edge-crossing expectations and can make the user label or cut the wrong frames.

**Fix:** Track whether the pointer moved. Apply the one-second fallback only to a no-movement click in `new` mode; preserve a deliberate one-frame drag/resize, with a one-frame minimum on the side selected after crossing.

### 2. Starting a new drag after Cancel or while a cut is open appends a selection element on every pointer move

**Evidence:** `src/dataop_logger/viewer.js:460,468-470`; Cancel clears the selection at `src/dataop_logger/viewer.js:381-382`, and an open cut renders no `#selbox` at `src/dataop_logger/viewer.js:323-324`.

`box` is a constant initialized to `null`. In `draw()`, the newly appended element is never assigned back to it, so every `pointermove` appends another `id="selbox"`. A normal drag after pressing Cancel therefore leaves a visible trail and can create hundreds of duplicate-ID nodes until pointer-up rerenders the page.

**Fix:** Make `box` mutable and create/assign it once (`box ||= ...`). Add a browser-level regression test for Cancel -> drag and open-cut -> drag.

### 3. Near the end of a run, Edit selects the preceding two seconds instead of starting at the playhead

**Evidence:** `src/dataop_logger/viewer.js:247-250`

`Math.min(r.length - w, f)` shifts the start backward to preserve a full two-second width. For a playhead at 9.5 s in a 10 s run, Edit opens approximately `[8, 10)` rather than `[9.5, 10)`. That contradicts the stated “starting at the playhead” behavior and is especially risky when someone parks on a failure near the end before labeling it.

**Fix:** Clamp the start to the playhead (`lo = min(f, length - 1)`) and truncate `hi` at `length`. Use a one-frame minimum for very short or sub-Hz runs rather than forcing two frames.

### 4. The selection overlay makes labels and cuts underneath it unclickable on the timeline

**Evidence:** `src/dataop_logger/viewer.html:117-125,153-155`; `src/dataop_logger/viewer.js:313-324,450-453`

`.selbox` covers the full 76 px editing area at `z-index: 2`; labels and cuts have no stacking level above it. Since Edit immediately opens a selection, a label or cut inside that span receives the selection element as the pointer target, so the label/cut opening branches never run. Users can still find entries in the cards below, but the direct timeline interaction appears broken.

**Fix:** Put label and cut hit targets above the selection body while keeping resize handles above them, or limit the move hit surface to lanes that do not contain interactive labels/cuts. Add a test that opens a label and a cut overlapped by the current selection.

## Low

### 5. The new scrub and selection controls are pointer-only and have very small hit targets

**Evidence:** `src/dataop_logger/viewer.js:312-326`; `src/dataop_logger/viewer.html:126-132`

The 16 px ruler and 10 px resize handles are plain `div`/`span` elements with no focus or ARIA semantics. Keyboard arrows control the playhead through the enclosing timeline, but there is no keyboard path to focus, move, or resize the selection, and the ruler lacks `touch-action: none`. This is difficult for touch and keyboard users, particularly when both handles overlap on a short existing label.

**Fix:** Give the scrubber and selection/handles explicit accessible controls and keyboard behavior, expose their values with ARIA, enlarge the effective hit areas without enlarging the visuals, and disable browser touch gestures during ruler scrubbing.
