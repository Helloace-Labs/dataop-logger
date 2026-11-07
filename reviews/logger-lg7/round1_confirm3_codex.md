approve

No findings. Commit `b1b4617` correctly completes the fix.

`tests/export_baseline.py` `_mask()` preserves nested dictionaries, keys, lists and nulls. It masks only non-null leaf values under volatile keys. Stable fields remain compared.

`tests/data/export_default_baseline.json` updates all ten report and record source entries across five recipes. Each retains `data`, `meta` and `videos`, with `videos` null. No other baseline values changed.

No adjacent regression was found by inspection. No tests or scripts were run. Baseline regeneration from `main` was documented but not independently reproduced.