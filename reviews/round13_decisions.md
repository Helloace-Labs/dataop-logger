# Round 13 decisions

All findings were accepted and fixed. 64 tests pass. A browser check covered the real example and crafted edge cases: blend followed by expert merges into one intervention, a takeover request is shown as the switch's reason, a pre-recording event shows as "before", and a run starting under rescue has no false takeover.

| Finding | Decision |
| --- | --- |
| M1 "Rescued" hid the real result | The outcome always comes from the effective termination reason and failure mode. Any intervention adds a separate "rescue used" tag. |
| M2 takeover spans treated as separate interventions | `interventions()` merges contiguous expert/blend segments into one intervention. A handback appears only when policy control resumes. Expert or blend at frame 0 is the initial state, not a takeover. Every intervention after frame 0 gets a pre-takeover window, clipped at the start of the run. Header stats and the run card count interventions. |
| M3 takeover reasons and unfinished steps dropped | A recorded takeover event up to one second before a switch becomes that switch's note, and other recorded events are listed as they are. Unfinished steps appear as "unfinished at run end". |
| M4 data check hid zeros | All three checks always show. Latency carries a `measured` flag: equal frame and command times read "not measured", otherwise median and p95. |
| L1 nested sliders | The playhead ruler is the single focusable slider, with min, max, now and a visible focus ring. The selection's accessible name gives its frames and the `[` / `]` keys. |
| L2 pre-recording events | They keep frame -1, show "before" and sort first. Ties keep insertion order explicitly. |
| L3 robot and operator merged | They are separate rows. |
