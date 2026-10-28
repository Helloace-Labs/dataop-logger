# Round 9 decisions

All partly-fixed items are now closed. 59 tests pass, and the browser regression run is clean.

| Item | Decision |
| --- | --- |
| M1 touching stretches still cut the movement | Each stretch now leaves out the frame where the arm starts or stops moving (except at the very start or end of a run), so cutting every suggestion keeps every movement. The boundary tests assert this on the union of all suggestions. |
| L2 Cut lost the selection on failure | Cut restores the selection when the write fails, the same way saving a label does. |
| L4 falsey or out-of-limit readings accepted | Current and historical readings are validated with the same limits `interpret()` writes, including empty `{}` and empty strings. |
| L6 status precedence | Active cuts are checked first, so a span someone cut after keeping it reports `already_cut`. |
