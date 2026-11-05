request changes.

1. **High — Automatic imports guess that no policy or planner ran.**  
   Files: [importer.py](src/aftershock_logger/importer.py), `PRESETS`, `detect()` and `import_dataset()`; [cli.py](src/aftershock_logger/cli.py), import defaults.

   Default CLI imports select `demos` when intervention columns are missing. The new preset entries then write null policy and planner, including for policy recordings. Missing evidence does not establish that neither ran. The earlier fix still permits this through automatic selection.

   Fix: make the automatic fallback `plain`. Reserve null declarations for an explicitly selected `demos` preset or explicit mapping. Add coverage for automatic imports without intervention columns.

2. **High — Validation accepts policy evidence beside a null policy.**  
   Files: [validate.py](src/aftershock_logger/validate.py), `_check_run()`; [schema.py](src/aftershock_logger/schema.py), `manifest_problems()`; [test_planned_fields.py](tests/test_planned_fields.py), manifest validation tests.

   In a new manifest, changing `policy` to null still passes validation when ticks contain valid base actions or base/blend control. The writer and importer reject this contradiction, but validation checks only the policy’s type. Incorrect provenance can pass unnoticed.

   Fix: apply `manifest_fields()` to preserve legacy null handling. Reject an effective null policy when any tick has a valid base action or base/blend control. Add regression tests, including an expert frame with a shadow base action and a legacy null-policy manifest.

Static review only. No files were edited. No tests or application code were run.