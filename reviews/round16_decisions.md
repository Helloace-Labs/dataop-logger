# Round 16 decisions (functional test by Codex gpt-5.6-sol)

All findings were accepted and fixed, with tests.

| Finding | Decision |
| --- | --- |
| H1 remap dropped `lists.json` | `remap` copies `lists.json` and `batches.json` into the new root. If the destination already has a different file, it is kept and the report says to merge by hand. |
| H2 multi-field annotate saved a partial result | `annotate` builds and checks every requested change before writing any, so a bad field writes nothing. |
| M1 CLI lists had no describe or delete | Added `lists describe` and `lists delete` (delete still refuses an entry in use). |
| M2 idle detection assumed degrees | Each dimension is scaled by its own robust range in the run (2nd to 98th percentile), and a window is still when every dimension moves less than 2% of that. Units and action spaces no longer matter. Wording is "no movement" instead of "arm still". |
| L1 a no-op bumped the review revision | A transaction that changes nothing keeps its revision. |
