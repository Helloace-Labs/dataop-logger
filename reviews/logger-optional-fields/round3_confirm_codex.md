approve

1. **Import fix: correct and complete.** [importer.py](src/aftershock_logger/importer.py) removes null policy and planner declarations from every preset. Automatic imports retain `unknown`. Explicit mapping entries still permit null. No adjacent regression found.

2. **Validation fix: correct and complete.** [validate.py](src/aftershock_logger/validate.py) uses `manifest_fields()` and rejects an effective null policy when any tick has a valid base action or base/blend control. Legacy null policies still read as `unknown`. No adjacent regression found.

[test_planned_fields.py](tests/test_planned_fields.py) adds regression coverage for both fixes, including shadow base actions and legacy handling.

No High, Medium or Low findings. Static review only; no files changed, tests or application code run.