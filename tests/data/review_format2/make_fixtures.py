"""Write expected.json from the reference implementation of review format 2.

    AFTERSHOCK_CONTRACTS=<checkout with the contracts package> python tests/data/review_format2/make_fixtures.py

expected.json holds, for every case in tests/review_cases.py, the error, returned id, rev and sha256 of the canonical
text after each operation, and for every format 1 fixture the format 2 text it migrates to and the format 1 text it
goes back to, and for every file in l8_files.json (written by the logger at 4f4395e) the text the reference reads
it as, its digest and the text after the next change. The logger's tests compare against it without needing the reference implementation.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # tests/
sys.path.insert(0, str(HERE.parents[2] / "src"))

import review_cases as rc  # noqa: E402


def main() -> int:
    if not os.environ.get("AFTERSHOCK_CONTRACTS"):
        print("set AFTERSHOCK_CONTRACTS", file=sys.stderr)
        return 2
    impl = rc.ContractImpl(os.environ["AFTERSHOCK_CONTRACTS"])
    out = {"cases": {name: rc.run_case(impl, name)["steps"] for name in rc.CASES},
           "migration": rc.migration_results(impl), "l8_files": rc.l8_results(impl)}
    (HERE / "expected.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(out['cases'])} cases and {len(out['migration'])} migrations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
