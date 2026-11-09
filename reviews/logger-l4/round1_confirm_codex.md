Approve. All six items are resolved or correctly rejected. No adjacent regressions found in `git diff HEAD~1 HEAD`.

1. **High — Sync check: correct and complete.** `src/aftershock_logger/metadata.py:507` checks editability again under the dataset lock, before changing metadata.

2. **High — Metadata links: correct and complete.** `load`, validation and `_save` reject both valid and dangling links. This covers local writes, synced saves and export. Batch following uses `lexists` and skips invalid records.

3. **High — Rejection stands.** The supplied route text permits an unnamed proposal to return an existing detection unchanged in every status. `propose` preserves the record, revision and decision. Named suggestions still work.

4. **High — Size limit: correct and complete.** Both writers use `_save`, which checks the serialized byte count before replacing the file. Serialization and atomic replacement retain their previous behavior.

5. **Medium — Batch derivation: correct and complete.** Both metadata writers mark the affected batch for derivation within the same transaction. Stored `mixed_policy` updates even for locked batches. Locked membership stays fixed.

6. **Medium — Hash uniqueness: correct and complete.** Only string hashes reach the uniqueness check. Other types still produce validation errors, allowing validation to continue.

Read-only review. No tests or code were run.