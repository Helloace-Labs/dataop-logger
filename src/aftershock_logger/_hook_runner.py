"""Runs one export hook inside the sandbox. Started by hook_sandbox.py, never imported by the logger.

    python -I -S _hook_runner.py <job folder> <numpy package folder> <limits json>

It uses only the standard library and numpy, because nothing else is readable inside the sandbox. In order it sets
hard resource limits (the hook cannot raise them again), closes every inherited file descriptor above stderr, loads
numpy from its package folder, reads the run from `<job>/in` (JSON and .npy files, never pickle), loads the hook file
`<job>/in/hook.py`, calls the hook function and writes what it returned to `<job>/out` as .npy and JSON files.

Exit codes: 0 done, 2 the runner could not start (a limit could not be set, numpy would not load), 3 the hook raised
or returned something that is not a result, 4 probe mode only (see hook_sandbox.py)."""
import json
import os
import sys


def _limits(lim: dict) -> list:
    import resource
    done = []
    for name, value in lim.items():
        r = getattr(resource, name)
        try:
            resource.setrlimit(r, (value, value))
            done.append(name)
        except (ValueError, OSError):
            if name != "RLIMIT_AS":                   # macOS refuses RLIMIT_AS; the parent watches memory there
                raise
    return done


def _numpy(np_dir: str):
    import importlib.util
    sp = importlib.util.spec_from_file_location("numpy", os.path.join(np_dir, "__init__.py"),
                                                submodule_search_locations=[np_dir])
    mod = importlib.util.module_from_spec(sp)
    sys.modules["numpy"] = mod
    threads = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    for k in threads:                                 # one BLAS thread keeps memory small under RLIMIT_AS
        os.environ[k] = "1"
    try:
        sp.loader.exec_module(mod)
    finally:
        for k in threads:                             # the hook still sees an empty environment
            os.environ.pop(k, None)
    return mod


def _other_process_info(pid: int) -> str:
    """Try to read another process's command line and environment: /proc on Linux, KERN_PROCARGS2 on macOS."""
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                return "allowed" if f.read(1) else "denied: empty"
        except OSError as e:
            return f"denied: {e.errno}"
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        mib = (ctypes.c_int * 3)(1, 49, pid)                 # CTL_KERN, KERN_PROCARGS2
        size = ctypes.c_size_t(1 << 16)
        buf = ctypes.create_string_buffer(1 << 16)
        if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) == 0 and size.value > 8:
            return "allowed"
        return f"denied: {ctypes.get_errno()}"
    except Exception as e:                                  # noqa: BLE001 - no ctypes means no way in either
        return f"denied: {type(e).__name__}"


def _probe(job: str, done: list) -> None:
    """Try what a hostile hook would try and report each result. The parent decides whether the sandbox holds."""
    import socket
    out = {"env": sorted(os.environ), "limits": done}
    try:
        with open(os.path.join(job, "canary"), "rb") as f:
            f.read(1)
        out["read_outside"] = "allowed"
    except OSError as e:
        out["read_outside"] = f"denied: {e.errno}"
    wrote, errs = [], []
    for target in (os.path.join(job, "written"), "/aftershock-probe", "/tmp/aftershock-probe",
                   "/dev/aftershock-probe", "/dev/shm/aftershock-probe"):
        try:                                          # beside the job, and every place a sandbox root could hold
            with open(target, "xb") as f:
                f.write(b"x")
            wrote.append(target)
        except OSError as e:
            errs.append(e.errno)
    out["write_outside"] = f"allowed: {wrote}" if wrote else f"denied: {errs}"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(("127.0.0.1", int(open(os.path.join(job, "in", "port")).read())))
        out["network"] = "allowed"
    except OSError as e:
        out["network"] = f"denied: {e.errno}"
    try:
        os.kill(int(open(os.path.join(job, "in", "parent")).read()), 0)
        out["other_process"] = "allowed"
    except OSError as e:
        out["other_process"] = f"denied: {e.errno}"
    rt = open(os.path.join(job, "in", "runtime_file")).read()
    if rt:
        try:
            with open(rt, "rb") as f:
                f.read(1)
            out["read_runtime_non_code"] = "allowed"
        except OSError as e:
            out["read_runtime_non_code"] = f"denied: {e.errno}"
    else:
        out["read_runtime_non_code"] = "denied: no such file in the runtime"
    try:
        with open(os.path.join(job, "out", "extra"), "wb") as f:
            f.write(b"x")
        out["write_new_output"] = "allowed"
    except OSError as e:
        out["write_new_output"] = f"denied: {e.errno}"
    try:
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        u.settimeout(2)
        u.connect(open(os.path.join(job, "in", "unix")).read())
        out["unix_socket"] = "allowed"
    except OSError as e:
        out["unix_socket"] = f"denied: {e.errno}"
    made = []
    for kind, make in (("unix", lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),
                       ("inet", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
                       ("pair", lambda: socket.socketpair()[0])):
        try:
            make().close()
            made.append(kind)
        except OSError:
            pass
    out["socket_create"] = f"allowed: {made}" if made else "denied"
    try:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        out["fork"] = "allowed"
    except OSError as e:
        out["fork"] = f"denied: {e.errno}"
    out["other_process_info"] = _other_process_info(int(open(os.path.join(job, "in", "parent")).read()))
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    os._exit(4)


def main() -> None:
    job, np_dir, lim = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    os.environ.pop("PWD", None)                       # bubblewrap sets PWD to the input folder after --clearenv
    try:
        done = _limits(lim)
        os.closerange(3, 1 << 16)
        np = _numpy(np_dir)
    except BaseException as e:                       # noqa: BLE001 - any start failure is reported, never ignored
        sys.stderr.write(f"runner could not start: {type(e).__name__}: {e}\n")
        os._exit(2)
    if os.path.exists(os.path.join(job, "in", "probe")):
        _probe(job, done)
    inp, out = os.path.join(job, "in"), os.path.join(job, "out")
    try:
        meta = json.load(open(os.path.join(inp, "run.json")))
        run = {"manifest": meta["manifest"], "effective": meta["effective"], "recipe": meta["recipe"],
               "ticks": {k: np.load(os.path.join(inp, f"tick_{k}.npy"), allow_pickle=False) for k in meta["ticks"]},
               "keep": np.load(os.path.join(inp, "keep.npy"), allow_pickle=False),
               "action": np.load(os.path.join(inp, "action.npy"), allow_pickle=False)}
        run["ticks"]["mode"] = np.asarray(meta["mode"], dtype=object)
    except BaseException as e:                       # noqa: BLE001
        sys.stderr.write(f"runner could not read its inputs: {type(e).__name__}: {e}\n")
        os._exit(2)
    try:
        import importlib.util
        sp = importlib.util.spec_from_file_location("export_hook", os.path.join(inp, "hook.py"))
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        fn = getattr(mod, meta["function"], None)
        if not callable(fn):
            raise TypeError(f"the hook file has no function {meta['function']!r}")
        res = fn(run)
        if res is None:
            res = {}
        if not isinstance(res, dict):
            raise TypeError(f"the hook returned {type(res).__name__}, not a dict")
        extra = set(res) - {"keep", "action", "reason"}
        if extra:
            raise TypeError(f"the hook returned unknown keys {sorted(map(str, extra))}")
        if res.get("reason") is None:
            res.pop("reason", None)
        info = {"keys": sorted(res)}
        if "reason" in res:
            if not isinstance(res["reason"], str):
                raise TypeError("the hook's reason is not a string")
            info["reason"] = res["reason"]
        if "keep" in res:
            np.save(os.path.join(out, "keep.npy"), np.asarray(res["keep"]), allow_pickle=False)
        if "action" in res:
            a = np.asarray(res["action"])
            if a.dtype.kind == "f":                   # any float width is written as float32, as before
                a = a.astype(np.float32)
            np.save(os.path.join(out, "action.npy"), a, allow_pickle=False)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(info, f)
    except BaseException as e:                       # noqa: BLE001 - the parent turns this into not_run
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        where = next((f"line {fr.lineno}" for fr in reversed(tb) if fr.filename.endswith("hook.py")), "")
        sys.stderr.write(f"{type(e).__name__}: {e}" + (f" ({where} of the hook)" if where else "") + "\n")
        sys.stderr.flush()
        os._exit(3)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
