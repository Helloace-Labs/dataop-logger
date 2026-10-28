# Instructions for coding agents

**Adding the logger to a user's robot project:** follow [skills/dataop-logger/SKILL.md](skills/dataop-logger/SKILL.md). It covers checking the project fits, installing, changing the recording loop and verifying with `dataop-logger validate`.

**Changing this repo:** the tests need `lerobot` 0.6.x in the environment. Run `python -m pytest -q tests` and keep them passing. `adapter.py` is the only file that touches LeRobot internals, so keep version-specific code there. Design decisions and invariants are in [SPEC.md](SPEC.md).
