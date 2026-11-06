"""Export hooks run in a sandboxed child (logger task L6). Hostile hooks try the network, files outside their
folder, a planted secret in the home folder, the parent's environment, other processes, endless loops, memory and
huge output; every attempt must end as not_run (and fail the export), never as a crash or a silent pass."""
import json
import os
import platform
import socket
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import hook_sandbox as HS  # noqa: E402


def _available() -> str | None:
    try:
        return HS.sandbox_mode()
    except HS.SandboxUnavailable:
        return None


MODE = _available()
needs_sandbox = pytest.mark.skipif(MODE is None and platform.system() != "Darwin",
                                   reason="no hook sandbox on this machine (export refuses hooks here; see "
                                          "test_no_sandbox_means_export_refuses_the_hook)")


def test_macos_always_has_a_proven_sandbox():
    if platform.system() == "Darwin":
        assert MODE == "macos-seatbelt"


def _hook(tmp_path, body: str, name: str = "hook.py") -> str:
    p = tmp_path / name
    p.write_text("import numpy as np\n" + textwrap.dedent(body))
    return f"{p}:h"


def _run(n=10, dim=3) -> dict:
    return {"manifest": {"run_id": "r1", "length": n}, "effective": {"rejected": False}, "recipe": "dagger",
            "ticks": {"base": np.zeros((n, dim), np.float32), "expert": np.ones((n, dim), np.float32),
                      "sent": np.ones((n, dim), np.float32), "base_valid": np.ones(n, bool),
                      "expert_valid": np.ones(n, bool), "safety_modified": np.zeros(n, bool),
                      "mode": np.asarray(["expert"] * n, dtype=object)},
            "keep": np.ones(n, bool), "action": np.ones((n, dim), np.float32)}


def _call(spec, run=None, **lim):
    return HS.run_hook(HS.load(spec), run or _run(), HS.Limits(**lim) if lim else None)


@needs_sandbox
def test_a_good_hook_runs_and_its_result_is_checked(tmp_path):
    spec = _hook(tmp_path, """
        def h(run):
            keep = run['keep'].copy(); keep[5:] = False
            assert run['ticks']['mode'][0] == 'expert' and run['manifest']['run_id'] == 'r1'
            return {'keep': keep, 'action': run['ticks']['expert'] * 2.0, 'reason': 'first five'}
        """)
    r = _call(spec)
    assert r.status == "ok", r.reason
    assert r.keep.dtype == bool and r.keep.sum() == 5
    assert r.action.dtype == np.float32 and np.all(r.action == 2)
    assert r.hook_reason == "first five" and r.info["sandbox"] == MODE


@needs_sandbox
def test_hook_sees_none_of_the_parents_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "sk-not-a-real-key")
    spec = _hook(tmp_path, """
        import os
        def h(run):
            return {'reason': 'env=' + ','.join(sorted(os.environ))}
        """)
    r = _call(spec)
    assert r.status == "ok", r.reason
    assert r.hook_reason == "env="


@needs_sandbox
def test_hook_that_opens_a_socket_is_not_run(tmp_path):
    lsn = socket.socket()
    lsn.bind(("127.0.0.1", 0))
    lsn.listen(1)
    lsn.settimeout(0)
    port = lsn.getsockname()[1]
    try:
        spec = _hook(tmp_path, f"""
            import socket
            def h(run):
                s = socket.socket(); s.settimeout(3); s.connect(('127.0.0.1', {port}))
                return {{}}
            """)
        r = _call(spec)
        assert r.status == "not_run" and ("PermissionError" in r.reason or "Operation not permitted" in r.reason), r.reason
        with pytest.raises(BlockingIOError):
            lsn.accept()                                       # nothing ever reached the parent
        spec = _hook(tmp_path, """
            import urllib.request
            def h(run):
                urllib.request.urlopen('http://example.com', timeout=3)
            """, "web.py")
        assert _call(spec).status == "not_run"
    finally:
        lsn.close()


@needs_sandbox
def test_hook_cannot_read_a_planted_secret_or_anything_outside_its_inputs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    secret = home / ".config" / "fake-app" / "credentials.env"
    secret.parent.mkdir(parents=True)
    secret.write_text("FAKE_TOKEN=not-a-real-token\n")
    monkeypatch.setenv("HOME", str(home))
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    (hook_dir / "neighbour.txt").write_text("next to the hook")
    for target in (secret, hook_dir / "neighbour.txt", Path("/etc/hosts"), Path(sys.prefix) / "pyvenv.cfg",
                   Path(__file__)):
        spec = _hook(hook_dir, f"""
            def h(run):
                return {{'reason': open({str(target)!r}).read()[:20]}}
            """)
        r = _call(spec)
        assert r.status == "not_run", (target.name, r.reason)
        assert "not-a-real-token" not in r.reason + r.stderr
    spec = _hook(hook_dir, f"""
        import os
        def h(run):
            return {{'reason': ','.join(os.listdir({str(home)!r}))}}
        """)
    assert _call(spec).status == "not_run"


@needs_sandbox
def test_hook_cannot_write_outside_or_start_or_signal_processes(tmp_path):
    target = tmp_path / "pwned.txt"
    spec = _hook(tmp_path, f"""
        def h(run):
            open({str(target)!r}, 'w').write('x')
        """)
    assert _call(spec).status == "not_run" and not target.exists()
    spec = _hook(tmp_path, f"""
        import subprocess
        def h(run):
            subprocess.run(['/bin/sh', '-c', 'touch {target}'])
        """, "sub.py")
    assert _call(spec).status == "not_run" and not target.exists()
    spec = _hook(tmp_path, f"""
        import os, signal
        def h(run):
            os.kill({os.getpid()}, signal.SIGTERM)
        """, "sig.py")
    assert _call(spec).status == "not_run"                      # and this test process is still alive


@needs_sandbox
@pytest.mark.parametrize("body,why", [
    ("import time\ndef h(run):\n    time.sleep(30)\n", "timed out"),
    ("def h(run):\n    while True:\n        pass\n", "timed out"),
])
def test_hook_past_its_time_limit_is_stopped(tmp_path, body, why):
    t0 = time.monotonic()
    r = _call(_hook(tmp_path, body), timeout_s=2)
    assert r.status == "not_run" and why in r.reason
    assert time.monotonic() - t0 < 10


@needs_sandbox
def test_hook_over_its_memory_limit_is_stopped(tmp_path):
    spec = _hook(tmp_path, """
        def h(run):
            blocks = []
            for _ in range(64):
                blocks.append(np.ones(64 << 20, np.uint8))
        """)
    r = _call(spec, memory_mb=256, timeout_s=30)
    assert r.status == "not_run" and ("memory" in r.reason or "MemoryError" in r.reason), r.reason


@needs_sandbox
def test_hook_with_huge_output_is_stopped(tmp_path):
    spec = _hook(tmp_path, """
        import sys
        def h(run):
            while True:
                sys.stdout.write('x' * 65536)
        """)
    r = _call(spec, timeout_s=20)
    assert r.status == "not_run" and "printed more than" in r.reason
    assert len(r.stderr) <= 256 << 10
    spec = _hook(tmp_path, """
        def h(run):
            return {'keep': np.ones(10**9, bool)}
        """, "bigfile.py")
    r = _call(spec, timeout_s=30, memory_mb=2048)
    assert r.status == "not_run", r.reason


@needs_sandbox
def test_hook_cannot_write_a_huge_file_or_any_other_file(tmp_path):
    spec = _hook(tmp_path, """
        import os
        def h(run):
            out = os.path.join(os.path.dirname(__file__), '..', 'out')
            with open(os.path.join(out, 'keep.npy'), 'wb') as f:
                for _ in range(1024):
                    f.write(b'x' * (1 << 20))
        """)
    r = _call(spec, timeout_s=30)
    assert r.status == "not_run" and "File too large" in r.reason, r.reason
    spec = _hook(tmp_path, """
        import os
        def h(run):
            open(os.path.join(os.path.dirname(__file__), '..', 'out', 'extra.bin'), 'wb').write(b'x')
        """, "extra.py")
    r = _call(spec)
    assert r.status == "not_run", r.reason


@needs_sandbox
@pytest.mark.parametrize("body,want", [
    ("def h(run):\n    raise RuntimeError('boom')\n", "RuntimeError: boom"),
    ("def h(run):\n    return {'keep': np.ones(9, bool)}\n", "shape"),
    ("def h(run):\n    return {'keep': np.ones(10, np.int64)}\n", "dtype"),
    ("def h(run):\n    return {'action': np.ones((10, 2), np.float32)}\n", "shape"),
    ("def h(run):\n    return {'action': np.full((10, 3), np.nan)}\n", "non-finite"),
    ("def h(run):\n    return [1, 2]\n", "not a dict"),
    ("def h(run):\n    return {'kepe': 1}\n", "unknown keys"),
    ("import os\ndef h(run):\n    os._exit(0)\n", "no result file"),
    ("import os\ndef h(run):\n    os.symlink('/etc/hosts', os.path.join(os.path.dirname(__file__), '..', 'out', "
     "'result.json'))\n    os._exit(0)\n", "could not be opened safely"),
    ("import os\ndef h(run):\n    os.mkfifo(os.path.join(os.path.dirname(__file__), '..', 'out', 'result.json'))\n"
     "    os._exit(0)\n", "not a plain file"),
    ("import os, signal\ndef h(run):\n    os.kill(os.getpid(), signal.SIGSEGV)\n", "signal"),
    ("def g(run):\n    return {}\n", "no function"),
])
def test_a_failing_or_wrong_hook_is_not_run_with_the_reason(tmp_path, body, want):
    r = _call(_hook(tmp_path, body))
    assert r.status == "not_run" and want in r.reason, r.reason


def test_bad_hook_specs_are_refused_before_anything_runs(tmp_path):
    for spec in ("nofunction", f"{tmp_path}/missing.py:h", "no_such_module_xyz:h", f"{tmp_path}/x.py:not-a-name"):
        with pytest.raises(ValueError):
            HS.resolve(spec)


@needs_sandbox
def test_module_hook_is_found_without_running_it_in_this_process(tmp_path, monkeypatch):
    marker = tmp_path / "imported_here"
    pkg = tmp_path / "mods" / "myhooks"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    (pkg / "sel.py").write_text(f"open({str(marker)!r}, 'w').write('x')\ndef h(run):\n    return {{}}\n")
    monkeypatch.syspath_prepend(str(tmp_path / "mods"))
    path, fn = HS.resolve("myhooks.sel:h")
    assert path == (pkg / "sel.py").resolve() and fn == "h"
    assert not marker.exists()
    r = _call("myhooks.sel:h")
    assert r.status == "not_run" and not marker.exists()     # its write is refused inside the sandbox too


def test_no_sandbox_means_export_refuses_the_hook(tmp_path, monkeypatch):
    from test_logger import _locked_batch
    from aftershock_logger import export as X
    ds, ok, bad = _locked_batch(tmp_path)
    spec = _hook(tmp_path, "def h(run):\n    return {}\n")
    monkeypatch.setattr(HS, "_PROVEN", {"macos-seatbelt": "network (test)", "linux-bwrap": "network (test)"})
    with pytest.raises(X.ExportError, match="refuses to run the code hook"):
        X.export([(ds.root, "round 1")], "dagger", str(tmp_path / "out"), dry_run=True, hook=spec)
    monkeypatch.setattr(HS.platform, "system", lambda: "Plan9")
    with pytest.raises(X.ExportError, match="no sandbox for code hooks"):
        X.export([(ds.root, "round 1")], "dagger", str(tmp_path / "out"), dry_run=True, hook=spec)
    rep = X.export([(ds.root, "round 1")], "dagger", str(tmp_path / "out"), dry_run=True)     # no hook: unaffected
    assert rep["frames"] > 0


def test_a_failed_probe_refuses_hooks(monkeypatch):
    monkeypatch.setattr(HS, "_PROVEN", {})
    monkeypatch.setattr(HS, "_probe", lambda mode: "read_outside")
    with pytest.raises(HS.SandboxUnavailable, match="did not hold"):
        HS.sandbox_mode()


def test_python_inside_or_above_home_is_refused(monkeypatch):
    home = os.path.realpath(os.path.expanduser("~"))
    for p in ("/", home, os.path.dirname(home)):
        with pytest.raises(HS.SandboxUnavailable):
            HS._check_paths([p])
    with pytest.raises(HS.SandboxUnavailable):
        HS._check_paths(['/opt/py"(allow default)'])


@needs_sandbox
def test_export_fails_with_the_hook_name_and_records_the_sandbox(tmp_path):
    from test_logger import _locked_batch
    from aftershock_logger import export as X
    ds, ok, bad = _locked_batch(tmp_path)
    spec = _hook(tmp_path, "def h(run):\n    raise ValueError('nope')\n", "failing.py")
    with pytest.raises(X.ExportError, match=r"failing\.py:h did not run for run .*ValueError: nope"):
        X.export([(ds.root, "round 1")], "dagger", str(tmp_path / "a"), hook=spec)
    assert not (tmp_path / "a").exists()
    spec = _hook(tmp_path, "import time\ndef h(run):\n    time.sleep(10)\n", "slow.py")
    with pytest.raises(X.ExportError, match="timed out"):
        X.export([(ds.root, "round 1")], "dagger", str(tmp_path / "a"), dry_run=True, hook=spec, hook_timeout_s=1)
    spec = _hook(tmp_path, """
        def h(run):
            keep = run['keep'].copy(); keep[np.flatnonzero(keep)[2:]] = False
            return {'keep': keep, 'reason': 'two'}
        """, "good.py")
    rep = X.export([(ds.root, "round 1")], "dagger", str(tmp_path / "b"), hook=spec)
    rec = json.loads((tmp_path / "b" / "dataop_export.json").read_text())
    assert rec["hook"]["spec"] == spec and rec["hook"]["sandbox"] == MODE
    assert rec["hook"]["file_sha256"] == HS.load(spec).file_sha256
    assert rec["hook"]["limits"]["timeout_s"] == 60 and rec["hook"]["results_sha256"]
    assert rep["frames"] == 2 and rep["hook"]["sandbox"] == MODE


@pytest.mark.skipif(platform.system() != "Darwin", reason="weakens the macOS profile")
@pytest.mark.parametrize("weaken,caught", [
    ("(allow network*)", "network"),
    ("(allow file-read*)", "read_outside"),
    ("(allow file-write*)", "write_outside"),
    ("(allow process-info*)", "other_process_info"),
])
def test_the_probe_catches_a_weakened_sandbox(monkeypatch, weaken, caught):
    real = HS._seatbelt_profile
    monkeypatch.setattr(HS, "_seatbelt_profile", lambda job, writable: real(job, writable) + weaken + "\n")
    monkeypatch.setattr(HS, "_PROVEN", {})
    with pytest.raises(HS.SandboxUnavailable, match=caught):
        HS.sandbox_mode()


@pytest.mark.skipif(platform.system() != "Darwin", reason="reads another process through a macOS sysctl")
def test_hook_cannot_read_another_processes_command_line_or_environment(tmp_path):
    spec = _hook(tmp_path, f"""
        import ctypes
        def h(run):
            libc = ctypes.CDLL(None, use_errno=True)
            mib = (ctypes.c_int * 3)(1, 49, {os.getpid()})
            size = ctypes.c_size_t(1 << 16); buf = ctypes.create_string_buffer(1 << 16)
            if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) == 0 and size.value > 8:
                return {{'reason': 'read it'}}
            raise PermissionError('denied')
        """)
    r = _call(spec)
    assert r.status == "not_run" and r.hook_reason is None, r.reason


def test_bwrap_command_isolates_everything_and_binds_only_the_output_writable(tmp_path):
    job = os.path.realpath(tmp_path / "job")
    os.makedirs(os.path.join(job, "in"))
    os.makedirs(os.path.join(job, "out"))
    cmd = HS._bwrap_cmd("/usr/bin/bwrap", job)
    for flag in ("--unshare-all", "--die-with-parent", "--new-session", "--clearenv"):
        assert flag in cmd
    assert "--share-net" not in cmd
    writable = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    assert writable == [os.path.join(job, "out")]
    ro = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--ro-bind"]
    home = os.path.realpath(os.path.expanduser("~"))
    assert os.path.join(job, "in") in ro and job not in ro
    assert all(p != "/" and not home.startswith(p.rstrip("/") + "/") and p != home for p in ro)
