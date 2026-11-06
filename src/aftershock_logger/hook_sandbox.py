"""Run export hooks (task code) in a sandboxed child process.

A hook is user code that changes which frames an export keeps or which action it writes. It never runs inside the
logger's own process. Each call starts a fresh child that has

  - an empty environment and no inherited file descriptors except a null stdin and two capped output pipes,
  - read access only to the code files (CODE_FILE) of the standard library and numpy, the interpreter and its shared
    library, its own copy of the hook file and its input folder, and write access only to the three result files
    the parent creates in its output folder,
  - no network or sockets, no way to start, signal or inspect another process, and
  - a wall clock time limit, a CPU time limit, a memory limit, a file size limit and an output limit.

Data goes in and out as JSON and .npy files in a private temp folder (never pickle). The parent checks every result
file's size, dtype and shape before it uses it.

Sandbox modes:

  macos-seatbelt   macOS `sandbox-exec` with a deny-by-default profile. RLIMIT_AS is not enforced on macOS, so the
                   parent reads the child's memory footprint every 20 ms and kills it over the limit.
  linux-bwrap      Linux bubblewrap: new user, mount, pid, network, ipc and uts namespaces, only the paths above
                   mounted read only (each runtime folder as a private copy of its code files, each needed
                   shared library as a single file, only the result files writable), and a seccomp filter that refuses fork, clone, socket and io_uring, so one process
                   holds all the memory and RLIMIT_AS bounds it.

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

CODE_FILE = r"\.(py|pyc|so|dylib)(\.[0-9]+)*$"             # the only files the child may read from the runtime


def _python() -> tuple[str, list[str], str]:
    """The real interpreter binary, the runtime folders the child needs, and numpy's package folder."""
    exe = os.path.realpath(sys.executable)
    np_dir = os.path.realpath(os.path.dirname(np.__file__))
    return exe, _code_dirs(), np_dir


def _runtime() -> dict:
    """What the interpreter and numpy need, and nothing else of the installation: the standard library folder
    (without its site-packages), numpy's package folder and its bundled libraries, the interpreter binary and its
    shared library. From the folders only code files (CODE_FILE) are readable."""
    import sysconfig
    stdlib = os.path.realpath(os.path.dirname(os.__file__))
    np_dir = os.path.realpath(os.path.dirname(np.__file__))
    libs = [os.path.realpath(p) for p in (np_dir + ".libs", os.path.join(os.path.dirname(np_dir), "numpy.libs"))
            if os.path.isdir(p)]
    files = [os.path.realpath(sys.executable)]
    libdir, ldlib = sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")
    if libdir and ldlib and os.path.isfile(os.path.join(libdir, ldlib)):
        files.append(os.path.realpath(os.path.join(libdir, ldlib)))
    return {"dirs": [stdlib, *sorted({np_dir, *libs} - {stdlib})], "files": sorted(set(files)),
            "hidden": [os.path.join(stdlib, "site-packages")]}


def _code_dirs() -> list[str]:
    return _runtime()["dirs"]


def _check_paths(paths: list[str]) -> None:
    home = os.path.realpath(os.path.expanduser("~"))
    for p in paths:
        if any(c in p for c in '"\\\n\0'):
            raise SandboxUnavailable(f"a path the sandbox would expose has a character it cannot quote: {p!r}")
        if p in ("/", "") or home == p or home.startswith(p.rstrip("/") + "/"):
            raise SandboxUnavailable(f"the Python runtime lives at {p}, which would expose the whole home folder or "
                                     "disk to the hook; install Python in its own folder")


def _readable(job: str) -> list[str]:
    """Every path the child can read anything from (folders are limited to code files, except its input folder)."""
    rt = _runtime()
    paths = sorted({*rt["dirs"], *rt["files"], os.path.join(job, "in")})
    _check_paths(paths)
    return paths


def _seatbelt_profile(job: str, writable: list[str]) -> str:
    q = lambda p: '"' + p + '"'                                   # paths were checked by _check_paths
    rt = _runtime()
    _readable(job)
    _check_paths(writable)
    def code(ds):
        return " ".join(f'(require-all (subpath {q(d)}) (regex #"{CODE_FILE}"))'
                        f" (require-all (subpath {q(d)}) (vnode-type DIRECTORY))" for d in ds)
    # The last matching rule wins. The standard library is granted first, then its site-packages is denied, then
    # numpy is granted, so a numpy installed inside that site-packages stays readable and nothing else there is.
    std, rest = rt["dirs"][:1], rt["dirs"][1:]
    return "\n".join([
        "(version 1)",
        "(deny default)",
        f"(allow process-exec (literal {q(os.path.realpath(sys.executable))}))",
        f"(allow file-read* {code(std)})",
        "(deny file-read* " + " ".join(f"(subpath {q(h)})" for h in rt["hidden"]) + ")",
        f"(allow file-read* {code(rest)} " + " ".join(f"(literal {q(f)})" for f in rt["files"]) + ")",
        f"(allow file-read* (subpath {q(os.path.join(job, 'in'))}))",
        # the root folder itself (the interpreter stats it at start) and three devices; the system libraries come
        # from the dyld shared cache, so nothing under /usr, /System, /Users or /private is readable
        '(allow file-read* (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))',
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


_MIRRORS: dict[tuple, dict[str, str]] = {}


def _code_files(dirs: list[str], hidden: list[str], d: str):
    """(relative folder, file) for every code file under runtime folder d: no symlinks, nothing in a hidden folder
    or in another runtime folder nested inside d (that one is mounted on its own)."""
    import re
    pat = re.compile(CODE_FILE)
    skip = [h for h in hidden if h == d or h.startswith(d + "/")] + [x for x in dirs if x != d and x.startswith(d + "/")]
    for root, subdirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        subdirs[:] = [x for x in subdirs if not os.path.islink(os.path.join(root, x))
                      and not any(os.path.join(root, x) == k or os.path.join(root, x).startswith(k + "/")
                                  for k in skip)]
        yield rel, None
        for f in files:
            src = os.path.join(root, f)
            if pat.search(f) and not os.path.islink(src) and os.path.isfile(src):
                yield rel, f


def _mirror(dirs: list[str], hidden: list[str], leaves: tuple = (), links: tuple = ()) -> dict[str, str]:
    """Linux mode: a private read-only copy of each runtime folder that holds only its code files (hard linked when
    possible, else copied) and none of the hidden folders, mounted over the original path. Built once per process;
    a file added to the runtime later is simply not in it. Empty folders stand in for runtime folders nested inside
    another (numpy inside site-packages), and empty files for the shared libraries in `leaves` that lie inside a
    runtime folder (a bundled library under its symlink name), so their own mounts have a place to go. Each
    (path, target) in `links` inside a runtime folder is a symlinked folder the loader passes through; it is
    recreated in the copy as the same symlink. A symlink exposes nothing: only mounted files can be reached."""
    import atexit
    key = (tuple(dirs), tuple(hidden), tuple(leaves), tuple(links))
    if key in _MIRRORS:
        return _MIRRORS[key]
    base = os.path.realpath(tempfile.mkdtemp(prefix="aftershock-hook-runtime-"))
    atexit.register(shutil.rmtree, base, True)
    out = {}
    for i, d in enumerate(dirs):
        dst = os.path.join(base, str(i))
        for rel, f in _code_files(dirs, hidden, d):
            if f is None:
                os.makedirs(os.path.join(dst, rel), 0o700, exist_ok=True)
                continue
            src, target = os.path.join(d, rel, f), os.path.join(dst, rel, f)
            try:
                os.link(src, target)
            except OSError:
                shutil.copy2(src, target)
        for x in dirs:                                         # mount points for nested runtime folders
            if x != d and x.startswith(d + "/"):
                os.makedirs(os.path.join(dst, os.path.relpath(x, d)), 0o700, exist_ok=True)
        for leaf in leaves:                                    # mount points for libraries inside this folder
            if leaf.startswith(d + "/") and not any(x != d and x.startswith(d + "/") and leaf.startswith(x + "/")
                                                    for x in dirs):
                t = os.path.join(dst, os.path.relpath(leaf, d))
                os.makedirs(os.path.dirname(t), 0o700, exist_ok=True)
                if not os.path.lexists(t):
                    os.close(os.open(t, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        for link, target in links:                             # symlinked folders on a library's path
            if _owner(dirs, link) == d:
                t = os.path.join(dst, os.path.relpath(link, d))
                os.makedirs(os.path.dirname(t), 0o700, exist_ok=True)
                if not os.path.lexists(t):
                    os.symlink(target, t)
        out[d] = dst
    _MIRRORS[key] = out
    return out


_LIBS: dict[tuple, list[str]] = {}


def _owner(dirs: list[str], path: str) -> str | None:
    """The innermost runtime folder that holds path, or None."""
    inside = [d for d in dirs if path.startswith(d + "/")]
    return max(inside, key=len) if inside else None


def _canon_lib(path: str) -> str:
    """Where a library is mounted: its folder resolved (no `..`, no symlinked folders), its own name kept, so an
    alias stays an alias. The mirror's mount points, the containment checks and the binds all use this path. The
    spelling ldd printed (the ELF interpreter's /lib64/..., say) keeps working through `_folder_links`."""
    return os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))


def _folder_links(paths: list[str]) -> list[tuple[str, str]]:
    """Every symlinked folder the kernel passes through when it opens one of paths (as ldd printed them), as
    (symlink path, its target), following `..` and the targets' own symlinked folders the way the kernel does.
    The sandbox recreates these symlinks so the loader's own spellings resolve to the canonical mounts."""
    out: dict[str, str] = {}

    def walk(path: str, depth: int) -> None:
        if depth > 40:                                      # the kernel's own symlink limit
            raise SandboxUnavailable(f"too many symlinks on the library path {path}")
        cur = "/"
        for comp in path.split("/")[1:-1]:                  # folders only; the last name stays an alias file
            if comp in ("", "."):
                continue
            if comp == "..":
                cur = os.path.dirname(cur)
                continue
            nxt = os.path.join(cur, comp)
            if os.path.islink(nxt):
                target = os.readlink(nxt)
                out[nxt] = target
                walk(os.path.join(os.path.dirname(nxt), target, "x"), depth + 1)
                cur = os.path.realpath(nxt)
            else:
                cur = nxt
    for p in paths:
        walk(p, 0)
    return sorted(out.items())


def _parse_ldd(text: str) -> list[str]:
    """Library paths from ldd output, spaces kept: `name => /path (0x...)` or `/path (0x...)`. Only existing
    regular files are kept, so a misread line can never turn into a folder mount. Paths keep ldd's spelling;
    `_bwrap_cmd` places each at `_canon_lib` and recreates the symlinked folders of the spelling."""
    import re
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\S+ => )?(/.*) \(0x[0-9a-fA-F]+\)\s*$", line)
        if m and os.path.isfile(m.group(1)):
            out.append(m.group(1))
    return out


def _shared_libs(binaries: list[str]) -> list[str]:
    """Linux mode: the shared libraries (and the loader) the interpreter and the runtime's extension modules need,
    from ldd. Only these library files are mounted, never whole library folders."""
    key = tuple(binaries)
    if key in _LIBS:
        return _LIBS[key]
    ldd = shutil.which("ldd")
    if not ldd:
        raise SandboxUnavailable("ldd is missing, so the libraries a hook needs cannot be listed")
    libs = set()
    for b in binaries:
        try:
            r = subprocess.run([ldd, b], capture_output=True, text=True, timeout=30, env={})
        except (OSError, subprocess.TimeoutExpired):
            continue
        libs.update(_parse_ldd(r.stdout))
    _check_paths(sorted(libs))
    _LIBS[key] = sorted(libs)
    return _LIBS[key]


# syscalls the Linux child may not make: new processes or threads, any socket, io_uring (which can open sockets)
_SECCOMP_ARCH = {"x86_64": (0xC000003E, {"socket": 41, "socketpair": 53, "clone": 56, "fork": 57, "vfork": 58,
                                         "io_uring_setup": 425, "clone3": 435}),
                 "aarch64": (0xC00000B7, {"socket": 198, "socketpair": 199, "clone": 220, "io_uring_setup": 425,
                                          "clone3": 435})}


def seccomp_program(machine: str | None = None) -> bytes:
    """A classic BPF seccomp filter: another architecture (or the x32 ABI) is killed, the syscalls above fail with
    EPERM, everything else is allowed. Raises SandboxUnavailable on an architecture it does not know."""
    import struct
    m = machine or platform.machine()
    m = {"amd64": "x86_64", "arm64": "aarch64"}.get(m.lower(), m.lower())
    if m not in _SECCOMP_ARCH:
        raise SandboxUnavailable(f"no syscall filter for the {m} architecture, so a code hook cannot be isolated")
    arch, calls = _SECCOMP_ARCH[m]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    ALLOW, KILL, EPERM = 0x7FFF0000, 0x80000000, 0x00050000 | 1
    nrs = sorted(calls.values())
    ins = [(LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, KILL), (LD, 0, 0, 0)]
    checks = ([(JGE, None, 0, 0x40000000)] if m == "x86_64" else []) + [(JEQ, None, 0, n) for n in nrs]
    n_checks = len(checks)
    for i, (code, _, jf, k) in enumerate(checks):
        ins.append((code, n_checks - i, jf, k))            # jump over the remaining checks and ALLOW to EPERM
    ins += [(RET, 0, 0, ALLOW), (RET, 0, 0, EPERM)]
    return b"".join(struct.pack("<HBBI", c, jt, jf, k) for c, jt, jf, k in ins)


def _bwrap_cmd(bwrap: str, job: str, seccomp_fd: int | None = None) -> list[str]:
    import re
    rt = _runtime()
    _readable(job)
    exe = os.path.realpath(sys.executable)
    pat = re.compile(r"\.so(\.[0-9]+)*$")
    exts = [os.path.join(d, rel, f) for d in rt["dirs"] for rel, f in _code_files(rt["dirs"], rt["hidden"], d)
            if f and pat.search(f)]
    raw = _shared_libs([exe, *rt["files"][1:], *exts])
    libs = sorted({_canon_lib(x) for x in raw})
    links = _folder_links(raw)
    _check_paths(libs)
    mirror = _mirror(rt["dirs"], rt["hidden"], tuple(sorted({*libs, *rt["files"]})), tuple(links))
    cmd = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--cap-drop", "ALL",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    if os.path.isfile("/etc/ld.so.cache"):                        # where the loader looks a library up
        cmd += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for d in sorted(rt["dirs"], key=len):                         # outer folders first, so nested mounts land inside
        cmd += ["--ro-bind", mirror[d], d]
    for link, target in links:                                    # symlinked folders outside the runtime copies
        if _owner(rt["dirs"], link) is None:
            cmd += ["--symlink", target, link]
    for f in rt["files"]:                                         # files after folders, so no folder hides them
        cmd += ["--ro-bind", f, f]
    for lib in libs:
        cmd += ["--ro-bind", os.path.realpath(lib), lib]
    inp, out = os.path.join(job, "in"), os.path.join(job, "out")
    _check_paths([inp, out])
    cmd += ["--ro-bind", inp, inp, "--ro-bind", out, out]         # no new files in the output folder
    for f in _RESULT_FILES:                                       # only the precreated result files are writable
        cmd += ["--bind", os.path.join(out, f), os.path.join(out, f)]
    if seccomp_fd is not None:
        cmd += ["--seccomp", str(seccomp_fd)]
    cmd += ["--chdir", inp, exe]
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


PROBES = ("read_outside", "read_runtime_non_code", "write_outside", "write_new_output", "network", "unix_socket",
          "fork", "other_process", "other_process_info")


def _judge(mode: str, got: dict) -> list[str]:
    """What the probe report shows got through. On Linux a socket must not even be created (the seccomp filter),
    because the parent's listeners are outside the child's mounts and a failed connect would prove nothing."""
    names = PROBES + (("socket_create",) if mode == "linux-bwrap" else ())
    bad = [k for k in names if not str(got.get(k, "")).startswith("denied")]
    if got.get("env"):
        bad.append(f"environment not empty ({len(got['env'])} variables)")
    want = {"RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE"} | ({"RLIMIT_AS"} if mode == "linux-bwrap" else set())
    if not want <= set(got.get("limits", [])):
        bad.append(f"limits not set: {sorted(want - set(got.get('limits', [])))}")
    return bad


def _runtime_non_code_sample() -> str | None:
    """A file in the runtime folders that is not code (a license or data file), to prove the child cannot read it."""
    import re
    pat = re.compile(CODE_FILE)
    for d in _runtime()["dirs"]:
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and not os.path.islink(p) and not pat.search(name):
                return p
    return None


def _probe(mode: str) -> str:
    """Run the runner in probe mode with no hook and report what got through, or "" when everything was denied."""
    job = os.path.realpath(tempfile.mkdtemp(prefix="aftershock-hook-probe-"))
    lsn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udir = os.path.realpath(tempfile.mkdtemp(prefix="ah-", dir="/tmp"))   # short: unix socket paths are limited
    ulsn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        ulsn.bind(os.path.join(udir, "s"))
        ulsn.listen(1)
        ulsn.settimeout(0)
        lsn.bind(("127.0.0.1", 0))
        lsn.listen(1)
        lsn.settimeout(0)
        _layout(job)
        Path(job, "canary").write_text("secret")
        Path(job, "in", "probe").write_text("1")
        Path(job, "in", "port").write_text(str(lsn.getsockname()[1]))
        Path(job, "in", "parent").write_text(str(os.getpid()))
        Path(job, "in", "unix").write_text(os.path.join(udir, "s"))
        nc = _runtime_non_code_sample()
        Path(job, "in", "runtime_file").write_text(nc or "")
        res = _spawn(mode, job, Limits(timeout_s=30, memory_mb=1024), file_bytes=1 << 20)
        if res.get("killed") or res["code"] != 4:
            return f"probe exited with {res['code']}: {res['stderr'][-300:].strip()}"
        try:
            got = json.loads(res["stdout"])
        except ValueError:
            return "probe printed no report"
        bad = _judge(mode, got)
        for name, s_ in (("network", lsn), ("unix_socket", ulsn)):
            try:
                s_.accept()[0].close()
                bad.append(f"{name} (connection reached the parent)")
            except (BlockingIOError, OSError):
                pass
        if sorted(os.listdir(os.path.join(job, "out"))) != sorted(_RESULT_FILES):
            bad.append("write_new_output (file created)")
        if Path(job, "written").exists():
            bad.append("write_outside (file created)")
        return "; ".join(bad)
    except SandboxUnavailable as e:
        return str(e)
    finally:
        lsn.close()
        ulsn.close()
        shutil.rmtree(job, ignore_errors=True)
        shutil.rmtree(udir, ignore_errors=True)


# ── one child ─────────────────────────────────────────────────────────────────

def _layout(job: str) -> None:
    os.mkdir(os.path.join(job, "in"), 0o700)
    os.mkdir(os.path.join(job, "out"), 0o700)
    shutil.copyfile(RUNNER, os.path.join(job, "in", "runner.py"))
    for f in _RESULT_FILES:                           # the child may write these and create nothing else
        os.close(os.open(os.path.join(job, "out", f), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))


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
        prog = seccomp_program()
        filt = os.path.join(job, "seccomp.bpf")
        Path(filt).write_bytes(prog)
        seccomp_fd = os.open(filt, os.O_RDONLY)
        cmd = [*_bwrap_cmd(shutil.which("bwrap"), job, seccomp_fd), *args]
    else:
        raise SandboxUnavailable(f"unknown sandbox mode {mode}")
    try:
        return _watch(mode, cmd, job, lim, () if mode != "linux-bwrap" else (seccomp_fd,))
    finally:
        if mode == "linux-bwrap":
            os.close(seccomp_fd)


def _watch(mode: str, cmd: list[str], job: str, lim: Limits, pass_fds: tuple) -> dict:
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
                         close_fds=True, pass_fds=pass_fds, start_new_session=True, cwd=os.path.join(job, "in"))
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
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        os.close(fd)
        raise ValueError(f"{what} is not a plain file")
    with io.FileIO(fd, "rb") as f:
        if st.st_size > limit:
            raise ValueError(f"{what} file is larger than {limit} bytes")
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{what} file is larger than {limit} bytes")
    return data


def _result_meta(out: Path) -> dict:
    """The child's result.json, checked against the full schema here in the parent, because the runner inside the
    sandbox shares its process with the hook and cannot be trusted after the hook ran:
    {"keys": [unique names from keep, action, reason], "reason": str when "reason" is in keys}. The output folder
    must hold exactly the precreated result files, and a result file whose key is not declared must be empty."""
    names = sorted(os.listdir(out))
    if names != sorted(_RESULT_FILES):
        raise ValueError(f"the output folder holds {names}, not just the result files")
    meta = json.loads(_read_result(out / "result.json", 64 << 10, "result"))
    if not isinstance(meta, dict) or not set(meta) <= {"keys", "reason"} or not isinstance(meta.get("keys"), list):
        raise ValueError("the result file is not a result")
    keys = meta["keys"]
    if not all(isinstance(k, str) for k in keys) or len(set(keys)) != len(keys) \
            or not set(keys) <= {"keep", "action", "reason"}:
        raise ValueError(f"the result names unknown or repeated keys {keys!r}"[:200])
    if ("reason" in keys) != ("reason" in meta) or ("reason" in meta and not isinstance(meta["reason"], str)):
        raise ValueError("the result's reason does not match its keys")
    for k, f in (("keep", "keep.npy"), ("action", "action.npy")):
        if k not in keys and _read_result(out / f, 0, k):
            raise ValueError(f"{f} was written but the result does not declare {k}")
    return meta


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
            meta = _result_meta(out)
            keys = set(meta["keys"])
            k2 = _load_npy(out / "keep.npy", bool, (n,), "keep") if "keep" in keys else None
            a2 = _load_npy(out / "action.npy", np.float32, action.shape, "action") if "action" in keys else None
            mask = k2 if k2 is not None else keep
            if a2 is not None and not np.all(np.isfinite(a2[mask])):
                raise ValueError("action has non-finite values on kept frames")
            reason = meta.get("reason")
            if reason is not None:
                reason = " ".join("".join(c if c.isprintable() else " " for c in str(reason)).split())[:200]
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
            return HookResult("not_run", f"bad result: {type(e).__name__ if isinstance(e, RecursionError) else e}", stderr=res["stderr"], info=info)
        return HookResult("ok", keep=k2, action=a2, hook_reason=reason, stderr=res["stderr"], info=info)
    except SandboxUnavailable as e:
        return HookResult("not_run", f"sandbox unavailable: {e}")
    except OSError as e:
        return HookResult("not_run", f"could not run the hook: {e}")
    finally:
        shutil.rmtree(job, ignore_errors=True)
