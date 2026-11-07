approve

No findings within the three fixes in `dec676b`.

1. **Prior High: auto-detection — correct and complete.**  
   `src/aftershock_logger/importer.py`, `detect()`: the HIL recording preset now requires the marker alongside both transition columns. Unmarked transition columns select `plain`. Earlier preset precedence and the demos fallback remain intact. Tests cover marked detection, explicit selection and ambiguous dense rewards.

2. **Prior High: finite threshold values — correct and complete.**  
   `src/aftershock_logger/importer.py`, `_build_run()`: only finite values establish success. Failure requires every value to be finite. Otherwise the outcome stays unknown with a diagnostic count. Both HIL presets receive the fix. Boolean, constant and file outcome mappings remain unchanged. New tests cover the reported invalid-value cases.

3. **Prior Low: field mapping table — correct and complete.**  
   `docs/DESIGN_batches_import_export.md`, section 2, “Field mapping”: the outcome row now describes custom `next.success` mapping on any frame and the HIL reward threshold above zero. It matches the implementation and changes no runtime behavior.

No nearby regressions found by inspection. Tests were read, not run. No files were changed.