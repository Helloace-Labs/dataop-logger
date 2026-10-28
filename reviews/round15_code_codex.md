# Round 15 code review

## Verdict

Numbering and grouping make ordinary in-run events much easier to match, and grouped titles are escaped correctly. The end event is retained and visually clamped at the right edge; the pin lane is vertically separated from cuts, selection handles, and the ruler. Two medium gaps remain: pre-recording rows have no corresponding pin, and pin buttons do not work from the keyboard or expose a meaningful screen-reader name. There are no high-severity findings. Tests: `64 passed in 5.10s`.

## High

None.

## Medium

### 1. Pre-recording events are numbered in the list but omitted from the pin lane

**Evidence:** `src/dataop_logger/viewer.js:404-419`

`numbered()` assigns frame `-1` events numbers, but `pinGroups()` filters every negative frame out. With a pre-recording event, the list begins at 1 while the first visible pin may be 2 or 3. This directly breaks the promised one-to-one correspondence and is confusing on first use.

**Fix:** Include pre-recording events in a left-edge group by mapping their pin position/seek frame to 0 while retaining “before recording” in the title and accessible label. Group them with frame-zero events when necessary, so one `1+` pin can highlight all corresponding rows.

### 2. Pins are tabbable but cannot be activated by keyboard and are announced only as their badge text

**Evidence:** `src/dataop_logger/viewer.js:321-322,542-545,592-600`

Seeking is implemented only in the timeline's `pointerdown` handler. Pressing Enter or Space on a focused pin produces a click, not a pointerdown, and the generic delegated click handler has no `data-pin` branch, so nothing happens. Hover linkage also has no focus/blur equivalent. Because the button's visible text is only `7` or `7+`, its multiline `title` is not a reliable accessible name; a screen reader cannot tell which events it represents.

**Fix:** Attach a normal `click` handler to pins (and avoid double-seeking from pointerdown), add focus/blur highlighting, and set an escaped `aria-label` containing the represented event numbers, times, and summaries. Add a keyboard test for Enter and Space on single and grouped pins.

## Low

### 1. Edge clamping assumes every pin is 18 px wide, and the playhead paints over unhighlighted pins

**Evidence:** `src/dataop_logger/viewer.js:321-322`; `src/dataop_logger/viewer.html:86-93`

The left/right clamp uses a fixed 9 px half-width, but multi-digit badges such as `100+` grow wider than 18 px and can overflow the timeline at frame 0 or `length`. Also, the playhead is `z-index: 3` while a normal pin is `z-index: 2`, so at the current event it draws through the badge number until hover raises the pin.

**Fix:** Give pins a known fixed width or clamp using their measured half-width, and keep the badge above the playhead (or stop the playhead line at the pin lane) while preserving pointer access.
