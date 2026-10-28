# Round 12 decisions

All findings were accepted and fixed. 64 tests pass. A browser check covered a drag after Cancel (one selection element), a near-one-frame resize kept as dragged, and Edit near the end of a run starting at the playhead.

| Finding | Decision |
| --- | --- |
| 1 one-frame resize expanded to a second | The one-second fallback applies only to a click (no movement) on the bar in "new" mode. Resizes and moves keep exactly what was dragged, with a one-frame minimum when an edge lands on the other. |
| 2 duplicate selection elements | `box` is created once and reused during the drag. |
| 3 default selection not at the playhead near the end | Edit starts the selection at the playhead and truncates it at the end of the run (a one-frame minimum). |
| 4 selection covered labels and cuts | The label lane and cuts sit above the selection. The resize handles start below the label lane, so labels and cuts stay clickable inside a selection. |
| 5 pointer-only, small targets | Handles have 16 px hit areas. The ruler is a labelled slider with `aria-valuenow` and `touch-action: none`. In Edit, `[` and `]` set the selection's start and end at the playhead, so a keyboard user can select frames with the arrow keys. |

The same commit includes the owner's redesign of the run details:
- **Run card:** one card answers what happened, what ran, who drove, under which conditions (the team's own fields) and whether the data can be trusted (control-loop latency, safety-filter changes, cut time). IDs move to one small line.
- **Major events:** Steps, Events and Labels are replaced by one list in time order. It holds the run's major events (steps done or failed, takeovers, handbacks, a marked failure start, cuts, suggested cuts, labels, the end). It also highlights the two seconds before each takeover, which is where the policy went wrong and what matters most for training data. The same window is shaded on the bar.
