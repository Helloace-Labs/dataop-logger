# Round 20 decisions

All five findings accepted and fixed in 571bacb.

- H1: export takes runs_by_root (resolved path -> run ids); the viewer sends picks per dataset; the writer matches snapshots by (root, run id). Test: two copies sharing run ids.
- M1: training_sets compares every recorded run in dataop_export.json against its own dataset, so a combined set shows out of date from every source. Test: edit one root, query both.
- M2: import_preview/import_start accept a mapping object (for agents, the --map format). The dialog stays click-only; when side files are not found it says which fields come in as unknown and how an agent can map them.
- M3: blend counts as the policy taking part (teleop/lora classification, and the run picker's "who drove"). Test: blend-only runs kept by dagger and lora.
- M4: CLI export accepts lora. Test added.
