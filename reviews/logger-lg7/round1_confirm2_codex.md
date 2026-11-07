request changes

**High — Nested source digest keys are still hidden.**

Files and sections: `tests/export_baseline.py`, `VARY` and `_mask()`; `tests/data/export_default_baseline.json`, every `sources[].lerobot`.

`_mask()` replaces the entire `lerobot` dictionary with `"<varies>"`. That dictionary contains `meta`, `data` and `videos`. Removing any of those keys, replacing the dictionary with `{}`, or changing `videos` from null would still pass the baseline comparison. The fix therefore remains incomplete.

Preserve the dictionary and every nested key. Mask only volatile digest values, retain nulls, and compare stable values. Regenerate the corrected baseline from `main`.

Run-record keys and stable source fields are now compared correctly. No adjacent runtime regression was found by inspection. Regeneration from `main` is documented in `reviews/logger-lg7/round1_decisions.md`; it was not independently reproduced. No tests or scripts were run.