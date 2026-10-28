# Round 14 decisions

All three partly fixed items are closed. 64 tests pass. Crafted browser cases check that: blend then expert then nobody reads "Shared control starts … Nobody in control"; a request carrying only a source keeps "requested by operator:alice at 0:00.8"; and a run starting under rescue has no false takeover.

| Item | Decision |
| --- | --- |
| M1 transitions misstated | Each intervention keeps the mode it started in and the modes before and after it. The start event uses the first mode. The end reads "Policy back in control" only when the policy resumes, and otherwise names who is in control. The pre-takeover window is shown only when the policy was in control just before. |
| M2 request source and time lost | A merged takeover request keeps its reason, text, requester and original time in the note. |
| M3 mixed timestamps | Latency percentiles use only frames with a positive frame-to-command time, and the page says how many frames were timed when not all were. |
