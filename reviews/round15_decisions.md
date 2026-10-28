# Round 15 decisions

All findings were accepted and fixed. A browser check covered pressing Enter on a focused pin (it seeks and lights its row), the pin's accessible name, and clicking the last pin.

| Finding | Decision |
| --- | --- |
| M1 pre-recording events had no pin | They join the left-edge pin, and their title and label read "before recording". |
| M2 pins not keyboard-usable or named | Pins seek from their own click handler, so Enter and Space work. Focus lights the matching rows, and each pin's `aria-label` lists its event numbers, times and text. |
| L1 clamping and paint order | Pins have a fixed 28 px width, are clamped by half of it, and sit above the playhead line. |
