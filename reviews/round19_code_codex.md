# Round 19 verification — `e0b33bc`

## Verdict

**Request two medium fixes.** The original export/unlock race, review reset, tree provenance, split records and dry-run nondeterminism are fixed. The exact commit passes **89 tests**. However, nested `batches.json` schema validation is incomplete, and the stale result of a multi-source export is registered inconsistently across its source batches.

I reviewed `git diff 5fba9d4..e0b33bc`, read `reviews/round18_decisions.md`, and tested an exact `git archive e0b33bc` with `/Users/chadster/lerobot/.venv/bin/python` and `PYTHONPATH=src`. An unrelated uncommitted `src/dataop_logger/importer.py` change appeared in the working tree during verification; it was left untouched and excluded by the archive test.

## Round 18 findings

| Finding | Status | Verification |
| --- | --- | --- |
| Export vs unlock race | **Fixed** | Barrier inside `save_episode`; unlocked and edited while blocked. Return value, batch registry and `dataop_export.json` were all `stale: true`. |
| Broken `batches.json` fails closed and validates | **Partly fixed** | Invalid JSON refuses review edits and `validate` reports it. Nested schema is still not validated safely; see Medium finding below. |
| Reviewed reset after an edit | **Fixed** | Marked reviewed, added an accepted exclusion: `reviewed` became false and normal lock was refused. |
| Complete output digest and records in both splits | **Fixed** | Recomputed each complete tree digest (excluding only the record) and matched `tree_sha256`; train and val both contain the same export ID and the correct `this_split`. Source digests now cover complete `meta`, `data` and `videos` trees. |
| Dry-run import IDs | **Fixed** | Two identical dry runs were byte-for-byte equal as sorted JSON; run IDs are `null`, not random UUIDs. |

## New findings

### Medium — Multi-source export can be current in one registry and stale in another

`export()` registers sources one at a time (`export.py:433-439`), and each `add_export()` independently computes stale under only that source's lock (`batches.py:373-382`). I unlocked source B between registration of A and B. The output record and return value were stale, but the registries recorded the same export as:

```text
output stale=true; source A stale=false; source B stale=true
```

That makes source A's batch picker advertise a globally stale combined artifact as current. A failure on a later registration can likewise leave earlier registries changed without any output record.

Fix: add a multi-source registration operation that acquires all distinct `batches.lock` files in canonical path order, rechecks every planned lock, computes one global stale result, and writes that same result to every source registry. Write the output record only after registration succeeds, with cleanup/explicit recovery for partial write failures. Add an interleaving test with two roots.

### Medium — Malformed nested batch schema crashes `validate`

`_read()` validates only the outer object plus batch name/status (`batches.py:70-87`). Setting a locked batch's `locked` field to a string passes `_read()` and then crashes `validate`:

```text
AttributeError: 'str' object has no attribute 'get'
```

Fix: validate required batch, export, history and lock fields and their nested types in `_read()`, raising `BatchError`; make `check()` convert every malformed-registry case to validation errors. Add cases for a non-object `locked`, malformed lock runs, and missing/wrong-typed `exports`/`history`.

## Regression checks

- `39bc314`'s `--drop-features` behavior remains intact on top of the original export implementation: source/base reads, videos, feature compatibility and output schema all honor the option, and provenance records it. A real export dropping `observation.state` reloaded without that feature and with `action` intact.
- Exact-commit test command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r19_commit_e0b33bc/src /Users/chadster/lerobot/.venv/bin/python -m pytest -q -p no:cacheprovider /tmp/r19_commit_e0b33bc/tests` — **89 passed in 5.15s**.
