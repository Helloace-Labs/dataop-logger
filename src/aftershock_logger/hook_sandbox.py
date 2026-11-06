"""Run export hooks (task code) in a sandboxed child process.

A hook is user code that changes which frames an export keeps or which action it writes. It never runs inside the
logger's own process. Each call starts a fresh child that has

  - an empty environment and no inherited file descriptors except a null stdin and two capped output pipes,
  - read access only to the Python runtime, numpy, its own copy of the hook file and its input folder, and write
    access only to the result files in its output folder,
  - no network, no way to start or signal another process, and
  - a wall clock time limit, a CPU time limit, a memory limit, a file size limit and an output limit.

Data goes in and out as JSON and .npy files in a private temp folder (never pickle). The parent checks every result
file's size, dtype and shape before it uses it.

Sandbox modes:

  macos-seatbelt   macOS `sandbox-exec` with a deny-by-default profile. RLIMIT_AS is not enforced on macOS, so the
                   parent reads the child's memory footprint every 20 ms and kills it over the limit.
  linux-bwrap      Linux bubblewrap: new user, mount, pid, network, ipc and uts namespaces, only the paths above
                   mounted (read only except the output folder), RLIMIT_AS for memory.

Before a mode is used for the first time in a process, a probe child tries what a hostile hook would try (read and
write outside its folders, open a socket, signal the parent, see the parent's environment). If any of it works, or
the mode is not available, code hooks are refused: the export fails closed and says why.

Any failure of a hook (a crash, an exception, a timeout, a limit, a result of the wrong shape or type) gives a result
with status `not_run` and the reason. It is never raised past `run_hook` as a crash and never treated as a pass;
export turns it into an error that names the hook."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.machinery
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).with_name("_hook_runner.py")
POLL = 0.02


class SandboxUnavailable(RuntimeError):
    """This machine cannot run a code hook with every isolation the logger requires."""


@dataclass
class Limits:
    timeout_s: float = 60.0              # wall clock, per run
    memory_mb: int = 1024
    output_kb: int = 256                 # stdout and stderr together
    file_mb: int = 0                     # largest result file; 0 = sized from the run (arrays plus 1 MB)

    def as_dict(self) -> dict:
        return {"timeout_s": self.timeout_s, "memory_mb": self.memory_mb, "output_kb": self.output_kb,
                "file_mb": self.file_mb}


@dataclass
class HookResult:
    status: str                          # "ok" or "not_run"
    reason: str = ""
    keep: np.ndarray | None = None
    action: np.ndarray | None = None
    hook_reason: str | None = None
    stderr: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class Hook:
    spec: str
    function: str
    source: bytes                        # the exact bytes that run, read once
    file_sha256: str
    mode: str

    @property
    def name(self) -> str:
        return self.spec


# ── resolving a hook without running any of its code ─────────────────────────

def resolve(spec: str) -> tuple[Path, str]:
    """`path/to/file.py:function` or `module.name:function`. A module is found on sys.path without importing it or
    its parent packages, so no hook code runs in this process. Only that one file is visible inside the sandbox, so
    a hook must be a single file that imports nothing but the standard library and numpy."""
    mod, _, fn = spec.rpartition(":")
    if not mod or not fn or not fn.isidentifier():
        raise ValueError("a hook is module:function or path/to/file.py:function")
    if mod.endswith(".py"):
        path = Path(mod).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"hook file {path.name} not found")
        return path, fn
    search, found = None, None
    for part in mod.split("."):
        found = importlib.machinery.PathFinder.find_spec(part if found is None else f"{found.name}.{part}", search)
        if found is None:
            raise ValueError(f"hook module {mod!r} not found")
        search = found.submodule_search_locations
    if found is None or not found.origin or not found.origin.endswith(".py"):
        raise ValueError(f"hook module {mod!r} is not a Python source file")
    return Path(found.origin).resolve(), fn


def load(spec: str) -> Hook:
    """Read the hook file once and check this machine can sandbox it. Raises ValueError for a bad spec and
    SandboxUnavailable when it cannot be run safely here."""
    path, fn = resolve(spec)
    src = path.read_bytes()
    mode = sandbox_mode()
    return Hook(spec=spec, function=fn, source=src, file_sha256=hashlib.sha256(src).hexdigest(), mode=mode)


# ── what the child may read ───────────────────────────────────────────────────

def _python() -> tuple[str, list[str], str]:
    """The real interpreter binary, the folders the interpreter needs (its prefix), and numpy's package folder."""
    exe = os.path.realpath(sys.executable)
    prefixes = sorted({os.path.realpath(sys.base_prefix), os.path.realpath(sys.base_exec_prefix)})
    np_dir = os.path.realpath(os.path.dirname(np.__file__))
    return exe, prefixes, np_dir


def _check_paths(paths: list[str]) -> None:
    home = os.path.realpath(os.path.expanduser("~"))
    for p in paths:
        if any(c in p for c in '"\\\n\0'):
            raise SandboxUnavailable(f"a path the sandbox would expose has a character it cannot quote: {p!r}")
        if p in ("/", "") or home == p or home.startswith(p.rstrip("/") + "/"):
            raise SandboxUnavailable(f"the Python runtime lives at {p}, which would expose the whole home folder or "
                                     "disk to the hook; install Python in its own folder")


def _readable(job: str) -> list[str]:
    exe, prefixes, np_dir = _python()
    libs = [p for p in (np_dir + ".libs", os.path.join(os.path.dirname(np_dir), "numpy.libs")) if os.path.isdir(p)]
    paths = sorted({*prefixes, np_dir, *libs, os.path.join(job, "in")})
    if not any(exe.startswith(p + "/") for p in prefixes):
        paths.append(exe)
    _check_paths(paths)
    return paths


def _seatbelt_profile(job: str, writable: list[str]) -> str:
    q = lambda p: '"' + p + '"'                                   # paths were checked by _check_paths
    exe = _python()[0]
    reads = _readable(job)
    _check_paths(writable)
    return "\n".join([
        "(version 1)",
        "(deny default)",
        f"(allow process-exec (literal {q(exe)}))",
        "(allow file-read* " + " ".join(f"(subpath {q(p)})" for p in reads) + ")",
        # the dynamic loader, system libraries and the root folder itself; nothing under /Users, /private/etc, ...
        '(allow file-read* (literal "/") (subpath "/usr/lib") (subpath "/System") (subpath "/private/var/db/dyld")'
        ' (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))',
        "(allow file-write* (literal \"/dev/null\") " + " ".join(f"(literal {q(p)})" for p in writable) + ")",
        # hardware facts numpy reads and uname; not kern.proc* or kern.procargs*, which would list other processes
        # and their command lines
        '(allow sysctl-read (sysctl-name-prefix "hw.") (sysctl-name "kern.osrelease") (sysctl-name "kern.ostype")'
        ' (sysctl-name "kern.version") (sysctl-name "kern.osversion") (sysctl-name "kern.osproductversion")'
        ' (sysctl-name "kern.hostname"))',
        "(allow signal (target self))",
        # not covered by deny default: without this a process can read another process's command line and
        # environment (KERN_PROCARGS2) and its details (proc_pidinfo)
        "(deny process-info*)",
        "(allow process-info* (target self))",
    ]) + "\n"


def _bwrap_cmd(bwrap: str, job: str) -> list[str]:
    exe = _python()[0]
    cmd = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--cap-drop", "ALL",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for p in ("/usr", "/lib", "/lib64", "/lib32", "/bin", "/etc/ld.so.cache"):   # the loader and C library
        if os.path.exists(p):
            cmd += ["--ro-bind", p, p]
    for p in _readable(job):
        cmd += ["--ro-bind", p, p]
    out = os.path.join(job, "out")
    _check_paths([out])
    cmd += ["--bind", out, out, "--chdir", os.path.join(job, "in"), exe]
    return cmd


# ── choosing and proving a sandbox ─────────────────────────────────────────────

_PROVEN: dict[str, str] = {}             # mode -> "" when the probe passed, else why it failed


def sandbox_mode() -> str:
    """The sandbox this machine provides, proven by a probe. Raises SandboxUnavailable (fail closed) otherwise."""
    system = platform.system()
    if system == "Darwin":
        mode = "macos-seatbelt"
        if not os.access("/usr/bin/sandbox-exec", os.X_OK):
            raise SandboxUnavailable("macOS sandbox-exec is missing, so a code hook cannot be isolated")
    elif system == "Linux":
        mode = "linux-bwrap"
        if not shutil.which("bwrap"):
            raise SandboxUnavailable("bubblewrap (bwrap) is not installed, so a code hook cannot be isolated; "
                                     "install bubblewrap")
    else:
        raise SandboxUnavailable(f"no sandbox for code hooks on {system}")
    if mode not in _PROVEN:
        _PROVEN[mode] = _probe(mode)
    if _PROVEN[mode]:
        raise SandboxUnavailable(f"the {mode} sandbox did not hold in its self test ({_PROVEN[mode]}), so code hooks "
                                 "are refused")
    return mode


def _probe(mode: str) -> str:
    """Run the runner in probe mode with no hook and report what got through, or "" when everything was denied."""
    job = os.path.realpath(tempfile.mkdtemp(prefix="aftershock-hook-probe-"))
    lsn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lsn.bind(("127.0.0.1", 0))
        lsn.listen(1)
        lsn.settimeout(0)
        _layout(job)
        Path(job, "canary").write_text("secret")
        Path(job, "in", "probe").write_text("1")
        Path(job, "in", "port").write_text(str(lsn.getsockname()[1]))
        Path(job, "in", "parent").write_text(str(os.getpid()))
        res = _spawn(mode, job, Limits(timeout_s=30, memory_mb=1024), file_bytes=1 << 20)
        if res.get("killed") or res["code"] != 4:
            return f"probe exited with {res['code']}: {res['stderr'][-300:].strip()}"
        try:
            got = json.loads(res["stdout"])
        except ValueError:
            return "probe printed no report"
        bad = [k for k in ("read_outside", "write_outside", "network", "other_process", "other_process_info")
               if got.get(k) != "denied" and not str(got.get(k, "")).startswith("denied")]
        try:
            lsn.accept()[0].close()
            bad.append("network (connection reached the parent)")
        except (BlockingIOError, OSError):
            pass
        if Path(job, "written").exists():
            bad.append("write_outside (file created)")
        if got.get("env"):
            bad.append(f"environment not empty ({len(got['env'])} variables)")
        want = {"RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE"} | ({"RLIMIT_AS"} if mode == "linux-bwrap" else set())
        if not want <= set(got.get("limits", [])):
            bad.append(f"limits not set: {sorted(want - set(got.get('limits', [])))}")
        return "; ".join(bad)
    except SandboxUnavailable as e:
        return str(e)
    finally:
        lsn.close()
        shutil.rmtree(job, ignore_errors=True)


# ── one child ─────────────────────────────────────────────────────────────────

def _layout(job: str) -> None:
    os.mkdir(os.path.join(job, "in"), 0o700)
    os.mkdir(os.path.join(job, "out"), 0o700)
    shutil.copyfile(RUNNER, os.path.join(job, "in", "runner.py"))


def _footprint(pid: int) -> int | None:
    """Physical memory footprint of a macOS process in bytes (proc_pid_rusage, RUSAGE_INFO_V2), or None."""
    try:
        lib = _footprint.lib if hasattr(_footprint, "lib") else ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _footprint.lib = lib
        buf = (ctypes.c_uint64 * 64)()
        if lib.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
            return None
        return int(buf[9])                             # ri_phys_footprint
    except OSError:
        return None


_RESULT_FILES = ("keep.npy", "action.npy", "result.json")


def _spawn(mode: str, job: str, lim: Limits, file_bytes: int) -> dict:
    exe, _, np_dir = _python()
    rl = {"RLIMIT_CPU": int(lim.timeout_s) + 1, "RLIMIT_FSIZE": int(file_bytes), "RLIMIT_NOFILE": 64,
          "RLIMIT_CORE": 0, "RLIMIT_AS": int(lim.memory_mb) << 20}
    args = ["-I", "-S", "-B", os.path.join(job, "in", "runner.py"), job, np_dir, json.dumps(rl)]
    out_dir = os.path.join(job, "out")
    if mode == "macos-seatbelt":
        prof = _seatbelt_profile(job, [os.path.join(out_dir, f) for f in _RESULT_FILES])
        cmd = ["/usr/bin/sandbox-exec", "-p", prof, exe, *args]
    elif mode == "linux-bwrap":
        cmd = [*_bwrap_cmd(shutil.which("bwrap"), job), *args]
    else:
        raise SandboxUnavailable(f"unknown sandbox mode {mode}")
    cap = int(lim.output_kb) << 10
    bufs = {"stdout": bytearray(), "stderr": bytearray()}
    over = threading.Event()

    def drain(stream, buf):
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            if len(bufs["stdout"]) + len(bufs["stderr"]) + len(chunk) > cap:
                buf.extend(chunk[:max(0, cap - len(bufs["stdout"]) - len(bufs["stderr"]))])
                over.set()
                continue                               # keep draining so the child never blocks on a full pipe
            buf.extend(chunk)
        stream.close()

    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={},
                         close_fds=True, start_new_session=True, cwd=os.path.join(job, "in"))
    threads = [threading.Thread(target=drain, args=(p.stdout, bufs["stdout"]), daemon=True),
               threading.Thread(target=drain, args=(p.stderr, bufs["stderr"]), daemon=True)]
    for t in threads:
        t.start()
    killed, t0 = None, time.monotonic()
    try:
        while p.poll() is None:
            if time.monotonic() - t0 > lim.timeout_s:
                killed = f"timed out after {lim.timeout_s:g} s"
            elif over.is_set():
                killed = f"printed more than {lim.output_kb} KB"
            elif mode == "macos-seatbelt":
                fp = _footprint(p.pid)
                if fp is None and p.poll() is None:
                    killed = "its memory could not be measured"
                elif fp is not None and fp > (int(lim.memory_mb) << 20):
                    killed = f"used more than {lim.memory_mb} MB of memory"
            if mode == "linux-bwrap" and not killed:
                extra = [n for n in os.listdir(out_dir) if n not in _RESULT_FILES]
                if extra:
                    killed = "wrote files other than its results"
            if killed:
                break
            time.sleep(POLL)
    finally:
        if p.poll() is None:
            for kill in (lambda: os.killpg(p.pid, signal.SIGKILL), p.kill):
                try:
                    kill()
                except OSError:
                    pass
        p.wait()
        for t in threads:
            t.join(5)
    if not killed and over.is_set():
        killed = f"printed more than {lim.output_kb} KB"
    return {"code": p.returncode, "killed": killed, "stdout": bufs["stdout"].decode("utf-8", "replace"),
            "stderr": bufs["stderr"].decode("utf-8", "replace")}


def _tail(s: str, n: int = 400) -> str:
    s = "".join(c if c.isprintable() or c == "\n" else "?" for c in s).strip()
    lines = [x for x in s.splitlines() if x.strip()]
    return (lines[-1] if lines else "")[-n:]


def _read_result(path: Path, limit: int, what: str) -> bytes:
    """Bytes of one result file the child wrote. It must be a plain file with one link (no symlink, hard link to
    another file, fifo or device) and at most `limit` bytes, so a hook cannot make the parent read anything else."""
    import io
    import stat
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise ValueError(f"no {what} file") from None
    except OSError as e:
        raise ValueError(f"{what} file could not be opened safely ({e.strerror})") from None
    with io.FileIO(fd, "rb") as f:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError(f"{what} is not a plain file")
        if st.st_size > limit:
            raise ValueError(f"{what} file is larger than {limit} bytes")
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{what} file is larger than {limit} bytes")
    return data


def _load_npy(path: Path, dtype, shape: tuple, what: str) -> np.ndarray:
    import io
    want = np.dtype(dtype)
    data = _read_result(path, int(np.prod(shape)) * want.itemsize + 4096, what)
    f = io.BytesIO(data)
    version = np.lib.format.read_magic(f)
    read = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
    if read is None:
        raise ValueError(f"{what} uses an unknown .npy format {version}")
    got_shape, _fortran, got_dtype = read(f, max_header_size=4096)
    if got_dtype != want:
        raise ValueError(f"{what} has dtype {got_dtype}, expected {want}")
    if tuple(got_shape) != tuple(shape):
        raise ValueError(f"{what} has shape {tuple(got_shape)}, expected {tuple(shape)}")
    a = np.load(io.BytesIO(data), allow_pickle=False)
    if a.dtype != want or a.shape != tuple(shape):
        raise ValueError(f"{what} is not {want} of shape {tuple(shape)}")
    return np.ascontiguousarray(a)


def run_hook(hook: Hook, run: dict, limits: Limits | None = None) -> HookResult:
    """Call the hook for one run in a fresh sandboxed child. `run` has manifest, effective, recipe, ticks (arrays,
    with mode as strings), keep (bool[n]) and action (float32[n, dim]). Never raises for anything the hook does:
    every failure is a HookResult with status not_run and the reason."""
    lim = limits or Limits()
    keep, action = np.asarray(run["keep"], bool), np.asarray(run["action"], np.float32)
    n = keep.shape[0]
    file_bytes = (int(lim.file_mb) << 20) if lim.file_mb else action.nbytes + n + (1 << 20)
    try:
        mode = sandbox_mode()
    except SandboxUnavailable as e:
        return HookResult("not_run", f"sandbox unavailable: {e}")
    if mode != hook.mode:
        return HookResult("not_run", f"the hook was loaded for sandbox {hook.mode}, not {mode}")
    job = os.path.realpath(tempfile.mkdtemp(prefix="aftershock-hook-"))
    try:
        _layout(job)
        inp = Path(job, "in")
        (inp / "hook.py").write_bytes(hook.source)
        ticks = {k: np.asarray(v) for k, v in run["ticks"].items() if k != "mode"}
        for k, v in ticks.items():
            np.save(inp / f"tick_{k}.npy", v, allow_pickle=False)
        np.save(inp / "keep.npy", keep, allow_pickle=False)
        np.save(inp / "action.npy", action, allow_pickle=False)
        (inp / "run.json").write_text(json.dumps({
            "manifest": run["manifest"], "effective": run["effective"], "recipe": run["recipe"],
            "function": hook.function, "ticks": sorted(ticks),
            "mode": [str(x) for x in run["ticks"].get("mode", [])]}, default=str))
        res = _spawn(mode, job, lim, file_bytes)
        info = {"sandbox": mode, "exit_code": res["code"]}
        if res["killed"]:
            return HookResult("not_run", f"stopped: {res['killed']}", stderr=res["stderr"], info=info)
        if res["code"] != 0:
            why = _tail(res["stderr"]) or f"exit code {res['code']}"
            if res["code"] is not None and res["code"] < 0:
                why = f"killed by signal {-res['code']}" + (f": {why}" if _tail(res["stderr"]) else "")
            return HookResult("not_run", f"failed: {why}", stderr=res["stderr"], info=info)
        out = Path(job, "out")
        try:
            meta = json.loads(_read_result(out / "result.json", 64 << 10, "result"))
            if not isinstance(meta, dict) or not isinstance(meta.get("keys", []), list):
                raise ValueError("the result file is not a result")
            keys = set(meta.get("keys", []))
            k2 = _load_npy(out / "keep.npy", bool, (n,), "keep") if "keep" in keys else None
            a2 = _load_npy(out / "action.npy", np.float32, action.shape, "action") if "action" in keys else None
            mask = k2 if k2 is not None else keep
            if a2 is not None and not np.all(np.isfinite(a2[mask])):
                raise ValueError("action has non-finite values on kept frames")
            reason = meta.get("reason")
            if reason is not None:
                reason = " ".join("".join(c if c.isprintable() else " " for c in str(reason)).split())[:200]
        except (OSError, ValueError, KeyError, TypeError) as e:
            return HookResult("not_run", f"bad result: {e}", stderr=res["stderr"], info=info)
        return HookResult("ok", keep=k2, action=a2, hook_reason=reason, stderr=res["stderr"], info=info)
    except SandboxUnavailable as e:
        return HookResult("not_run", f"sandbox unavailable: {e}")
    except OSError as e:
        return HookResult("not_run", f"could not run the hook: {e}")
    finally:
        shutil.rmtree(job, ignore_errors=True)
