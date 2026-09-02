"""RunPod serverless handler — PREPROCESS endpoint ("prep kitchen", 2026-07-15).

Deliberately dumb: receives a fully-built ComfyUI graph (built client-side by
aivfx-pipeline/preprocess_graph.py), runs it, renames the outputs to exact final
names in INPUTS_DIR (the flat folder the render endpoint reads), returns the file
list. No graph logic lives here — recipe changes never require an image rebuild.

Job input:
  {"graph": {...},                          # ComfyUI API-format prompt graph
   "outputs": {"<needle>": "<final.mp4>"},  # substring -> exact final filename
   "timeout_s": 1500}                       # optional
  or {"debug": true}                        # env/dir sanity check, no GPU work
  or {"fetch": {"url": "...", "dest": "ComfyUI/models/sam3/sam3.pt"}}
                                            # download big files straight to the volume
                                            # (local S3 uploads cap at ~100MB — stage
                                            # large model weights through this op)

Output files land in INPUTS_DIR (default /runpod-volume/native-xdit/inputs) —
flat folder, collision-proof names are the CLIENT's job (bake the date into the
tag, e.g. DRIVING-D2-0715-BAKED.mp4). See PREPROCESS_ENDPOINT_SPEC.md.
"""
import hashlib
import json, os, shutil, subprocess, sys, threading, time
import urllib.error
import urllib.request

import runpod

# Container-start model wiring (host-NVMe store resolution + symlink tree +
# aux-ckpts volume persistence). Runs before anything serves; its
# PREP_HOSTSTORE_ACTIVE|ABSENT line is the log check that fast loads are on.
_T_IMPORT = time.time()  # boot timeline origin: the handler process is up

# Network-volume reads arrive over the network: sampling /proc/net/dev RX bytes
# once a second during boot measures the model-load mechanism directly (bytes
# and MB/s per warm-up node) instead of inferring it from wall-clock.
_NET = {"samples": [], "stop": False}
def _net_rx_bytes():
    tot = 0
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, data = line.split(":", 1)
            if name.strip() == "lo":
                continue
            tot += int(data.split()[0])
    return tot
def _comfy_rchar():
    """Bytes ComfyUI has read via read() so far (/proc/<pid>/io rchar): the
    direct measure of model-file reading even when the volume is host-mounted
    (network counters inside the container see nothing then)."""
    pid = _comfy.pid if (_comfy is not None and _comfy.poll() is None) else None
    if not pid:
        return None
    with open(f"/proc/{pid}/io") as f:
        for line in f:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    return None
def _net_sampler():
    while not _NET["stop"]:
        try:
            _NET["samples"].append((time.time(), _net_rx_bytes(), _comfy_rchar()))
        except Exception:
            pass
        time.sleep(1)
threading.Thread(target=_net_sampler, daemon=True).start()
def _net_window(t_a, t_b):
    """Bytes in [t_a, t_b] from two counters: container network RX (blind to a
    host-mounted volume) and ComfyUI's read() bytes (rchar: the real load
    progress), each with the peak 1 s rate (MB/s)."""
    pts = [q for q in _NET["samples"] if t_a - 1 <= q[0] <= t_b + 1]
    out = {"gb": None, "peak_mbs": None, "read_gb": None, "read_peak_mbs": None, "read_avg_mbs": None}
    if len(pts) < 2:
        return out
    out["gb"] = round((pts[-1][1] - pts[0][1]) / 2**30, 2)
    out["peak_mbs"] = round(max((q2[1] - q1[1]) / max(q2[0] - q1[0], 1e-6) for q1, q2 in zip(pts, pts[1:])) / 2**20)
    rp = [(q[0], q[2]) for q in pts if len(q) > 2 and q[2] is not None]
    if len(rp) >= 2:
        out["read_gb"] = round((rp[-1][1] - rp[0][1]) / 2**30, 2)
        out["read_peak_mbs"] = round(max((b2 - b1) / max(t2 - t1, 1e-6) for (t1, b1), (t2, b2) in zip(rp, rp[1:])) / 2**20)
        out["read_avg_mbs"] = round((rp[-1][1] - rp[0][1]) / max(rp[-1][0] - rp[0][0], 1e-6) / 2**20, 1)
    return out
import prep_setup
_SETUP_STATE = prep_setup.run()
# Contract §4 (pipeline-contract.md): on an endpoint that depends on the host
# NVMe model store, a worker that lands on a host WITHOUT the staged models
# must die HERE, loudly — not serve an empty model tree and fail minutes later
# inside the graph. Opt-in by env so volume-fallback endpoints keep booting.
if os.environ.get("PREP_REQUIRE_HOSTSTORE") == "1" and "PREP_HOSTSTORE_ACTIVE" not in _SETUP_STATE:
    print(f"FATAL: PREP_REQUIRE_HOSTSTORE=1 but no staged host store ({_SETUP_STATE})", flush=True)
    sys.exit(3)
_T_SETUP = time.time()

VOL        = "/runpod-volume"

# Boot timeline (cold-start decomposition; ledger #5 / capacity design). Filled
# in as the worker boots and reported in EVERY job result + debug ping, so the
# backend's per-leg telemetry records it on a cold worker's first job without
# anyone reading container logs. All seconds, measured in-process; the part
# before the process exists (placement, image pull) is RunPod's delayTime
# minus ready_after_import_s.
BOOT = {"import_epoch": round(_T_IMPORT, 3),
        "setup_s": round(_T_SETUP - _T_IMPORT, 1),
        "model_store": _SETUP_STATE,
        "warmup": "pending"}

_GPU_INFO = None
def _gpu_info():
    """Which GPU this worker landed on (name, total MiB) — reported in every
    job result so per-job cost/placement can be analyzed from telemetry
    (capacity-strategy design L8). Cached; nvidia-smi is always present."""
    global _GPU_INFO
    if _GPU_INFO is None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            _GPU_INFO = [{"name": l.split(",")[0].strip(), "mem_mib": int(float(l.split(",")[1]))} for l in out if "," in l] or None
        except Exception as e:
            _GPU_INFO = {"error": str(e)[:80]}
    return _GPU_INFO
COMFY_DIR  = os.environ.get("COMFY_DIR", "/opt/ComfyUI")
PREP_MODELS_ROOT = "/opt/prep_models"  # symlink tree -> host store or volume (prep_setup)

# ---- fine hardware telemetry (the render endpoint's 8-stream set, ported 2026-09-02) ----
# Two 720-frame prep jobs died with "[Errno 111] Connection refused" and NOTHING
# recorded memory, GPU or process state across the death — the prep image never
# shipped pod_telemetry.sh. The render image has written meminfo/psi/vmstat/
# procstate at 1 Hz and GPU at 5 Hz to the volume on every boot since June.
# Same script, same layout: <volume>/serverless_telemetry/<endpoint>/<worker>/hw/.
# Streams land on the VOLUME directly, so they survive the worker that wrote them.
ENDPOINT  = os.environ.get("RUNPOD_ENDPOINT_ID", "unknown-ep")
WORKER_ID = os.environ.get("RUNPOD_POD_ID", f"pid{os.getpid()}")
_HERE     = os.path.dirname(os.path.abspath(__file__))
_TELE_DIR = os.path.join(os.environ.get("TELE_DIR", f"{VOL}/serverless_telemetry"), ENDPOINT, WORKER_ID)

def _hwtele(action, name=""):
    """start | phase <name> | stop — crash-guarded, niced loggers, never blocks a job."""
    try:
        script = os.path.join(_HERE, "pod_telemetry.sh")
        if not os.path.exists(script):
            print("hwtele: pod_telemetry.sh missing at", script, flush=True); return
        os.makedirs(_TELE_DIR, exist_ok=True)
        env = dict(os.environ); env["TELE_DIR"] = _TELE_DIR
        subprocess.Popen(["bash", script, action, "hw"] + ([name] if name else []),
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("hwtele", action, "failed:", repr(e), flush=True)

_hwtele("start")
_hwtele("phase", "handler_import")


def _read_seq(path, limit_bytes=None, chunk=32 * 2**20):
    """Sequential read of a file with big chunks (what the network volume is
    good at). Returns bytes read, seconds, MB/s. Side effect: the file lands in
    the host page cache, so later lazy (mmap) loads hit RAM instead of the
    network — the render endpoint's proven shard-prewarm pattern."""
    n = 0
    t0 = time.time()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
            if limit_bytes and n >= limit_bytes:
                break
    dt = max(time.time() - t0, 1e-6)
    return {"path": path, "gb": round(n / 2**30, 2), "s": round(dt, 1), "mbs": round(n / dt / 2**20)}


def _read_ranged(path, streams=16, chunk=16 * 2**20, limit_bytes=None, offset=0):
    """Read a file region with `streams` threads doing pread() on interleaved
    chunks — parallel ranged reads are what a latency-bound network volume
    needs (measured 8/21: one stream = ~38 MB/s). Same page-cache side effect."""
    size = os.path.getsize(path)
    end = min(size, offset + limit_bytes) if limit_bytes else size
    total = max(0, end - offset)
    n_chunks = (total + chunk - 1) // chunk
    fd = os.open(path, os.O_RDONLY)
    done = [0]
    lock = threading.Lock()
    def worker(start_idx):
        got = 0
        for i in range(start_idx, n_chunks, streams):
            off = offset + i * chunk
            want = min(chunk, end - off)
            pos = 0
            while pos < want:
                b = os.pread(fd, want - pos, off + pos)
                if not b:
                    break
                pos += len(b)
            got += pos
        with lock:
            done[0] += got
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max_workers=streams) as ex:
            list(ex.map(worker, range(min(streams, max(n_chunks, 1)))))
    finally:
        os.close(fd)
    dt = max(time.time() - t0, 1e-6)
    return {"path": path, "gb": round(done[0] / 2**30, 2), "s": round(dt, 1), "mbs": round(done[0] / dt / 2**20), "streams": streams}


def _prewarm_models(root=PREP_MODELS_ROOT, threads=None, min_bytes=8 * 2**20):
    """Read every model file under the prep model tree sequentially into the
    page cache (several files in parallel). Returns per-file and total rates."""
    files = []
    for dp, _, fs in os.walk(root, followlinks=True):
        for fn in fs:
            fp = os.path.join(dp, fn)
            try:
                rp = os.path.realpath(fp)
                if os.path.isfile(rp) and os.path.getsize(rp) >= min_bytes:
                    files.append(rp)
            except OSError:
                pass
    # Pre-warm is a NETWORK-VOLUME remedy (lazy mmap loads at 4-22 MB/s). Files served from the
    # host NVMe model store load fast on their own — skip them (saves the boot ~27 GB of reads).
    hs = [f for f in files if f.startswith("/runpod/model-store/")]
    files = [f for f in files if not f.startswith("/runpod/model-store/")]
    files = sorted(set(files), key=lambda x: -os.path.getsize(x))
    streams = int(threads or os.environ.get("PREP_PREWARM_STREAMS", "16"))
    out = {"files": [], "n": len(files), "streams": streams, "skipped_hoststore": len(hs)}
    t0 = time.time()
    # files one after another, each with `streams` parallel ranged readers
    # (total concurrency stays bounded; the big files dominate anyway)
    for fp in files:
        out["files"].append(_read_ranged(fp, streams=streams))
    dt = max(time.time() - t0, 1e-6)
    tot = sum(r["gb"] for r in out["files"])
    out.update({"total_gb": round(tot, 2), "s": round(dt, 1), "mbs": round(tot * 1024 / dt)})
    return out

INPUTS_DIR = os.environ.get("INPUTS_DIR", f"{VOL}/native-xdit/inputs")
STAGE_DIR  = os.environ.get("STAGE_DIR", f"{VOL}/native-xdit/prep_stage")
EMP        = os.environ.get("EXTRA_MODEL_PATHS", "/opt/extra_model_paths.yaml")
PORT       = int(os.environ.get("COMFY_PORT", "8188"))
URL        = f"http://127.0.0.1:{PORT}"
LOG        = "/tmp/comfy_prep.log"

_comfy = None


def _tail(path, n=40):
    try:
        with open(path, errors="replace") as f:
            return "\n".join(f.read().splitlines()[-n:])
    except OSError:
        return "<no log>"


def _errs(path, n_ctx=20):
    """Error-focused log digest: traceback/error lines first, then the last lines."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return "<no log>"
    keys = ("Traceback", "Error", "ERROR", "error:", "ModuleNotFound", "ImportError",
            "AssertionError", "CUDA", "Killed", "Segmentation")
    hits = [l for l in lines if any(k in l for k in keys)]
    return "ERR LINES:\n" + "\n".join(hits[-30:]) + "\n--- LAST LINES:\n" + "\n".join(lines[-n_ctx:])


def _cgroup_mem():
    """Container memory facts, cgroup v2 then v1. `oom_kill` is the decisive one:
    non-zero means the KERNEL killed something in this container for RAM, which
    is invisible from inside the dead process and looks only like a refused
    connection from here."""
    out = {}
    for key, path in (
        ("current", "/sys/fs/cgroup/memory.current"),
        ("peak", "/sys/fs/cgroup/memory.peak"),
        ("max", "/sys/fs/cgroup/memory.max"),
        ("current_v1", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ("peak_v1", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
        ("max_v1", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            with open(path) as f:
                out[key] = f.read().strip()
        except OSError:
            pass
    for path in ("/sys/fs/cgroup/memory.events", "/sys/fs/cgroup/memory/memory.oom_control"):
        try:
            with open(path) as f:
                for line in f:
                    k, _, v = line.strip().partition(" ")
                    if "oom" in k:
                        out[k] = v
        except OSError:
            pass
    return out


def _diagnose(where):
    """Why did ComfyUI stop answering? Collected AT the moment of failure, because
    RunPod purges job status within minutes and worker stdout is console-only.
    Distinguishes: process still alive (hang) / exited with a code (crash — the log
    carries the traceback, e.g. a CUDA OOM) / killed with no trace (kernel OOM-kill,
    which shows up in cgroup oom_kill and nowhere else).
    Never raises: a diagnosis that fails must not replace the original error."""
    d = {"where": where}
    try:
        if _comfy is None:
            d["comfy"] = "never started"
        else:
            rc = _comfy.poll()
            d["comfy"] = "alive" if rc is None else f"exited rc={rc}"
            if rc is not None and rc < 0:
                d["comfy_signal"] = -rc  # negative rc = killed by that signal (9 = SIGKILL/OOM)
    except Exception as e:
        d["comfy"] = f"<poll failed: {e}>"
    try:
        d["mem"] = _cgroup_mem()
    except Exception as e:
        d["mem"] = f"<{e}>"
    try:
        import subprocess as _sp
        d["gpu"] = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
    except Exception as e:
        d["gpu"] = f"<{e}>"
    try:
        d["comfy_log"] = _errs(LOG)[-4000:]
    except Exception as e:
        d["comfy_log"] = f"<{e}>"
    # The full ComfyUI log is on the worker's ephemeral disk — copy it next to
    # the telemetry streams so it outlives the worker, and stamp the moment.
    try:
        os.makedirs(_TELE_DIR, exist_ok=True)
        dst = os.path.join(_TELE_DIR, f"comfy_prep_{int(time.time())}.log")
        shutil.copy(LOG, dst)
        d["comfy_log_saved"] = dst
    except Exception as e:
        d["comfy_log_saved"] = f"<{e}>"
    _hwtele("phase", f"comfy_unreachable_{where.strip('/').replace('/', '_')[:40]}")
    # The diagnosis itself goes to the volume too — the job result reaches the
    # backend's logs, but this must be readable with nothing but the volume.
    try:
        with open(os.path.join(_TELE_DIR, f"diagnosis_{int(time.time())}.json"), "w") as f:
            json.dump(d, f, default=str, indent=1)
    except Exception:
        pass
    return d


def _http(path, payload=None, timeout=30):
    req = urllib.request.Request(URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:3000]
        raise RuntimeError(f"ComfyUI {path} HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        # Connection refused / reset: ComfyUI is not answering. On its own this
        # says nothing about WHY (observed live 2026-09-02 on a 720-frame graph:
        # the whole record was "[Errno 111] Connection refused"), so attach the
        # state that discriminates the causes.
        raise RuntimeError(
            f"ComfyUI {path} unreachable: {e.reason} | diagnosis="
            + json.dumps(_diagnose(path), default=str)[:6000]
        )


_COMFY_LOCK = threading.Lock()
def ensure_comfy(deadline_s=240):
    """Boot ComfyUI once per worker; reuse the warm process across jobs.
    Serialized: the background warm-up and the first job may both call it."""
    with _COMFY_LOCK:
        return _ensure_comfy_locked(deadline_s)


def _ensure_comfy_locked(deadline_s):
    global _comfy
    if _comfy is not None and _comfy.poll() is None:
        return
    os.makedirs(INPUTS_DIR, exist_ok=True)
    os.makedirs(STAGE_DIR, exist_ok=True)
    # launch main.py directly with real argv — the runpy/COMFY_ARGS trick silently
    # fails on current ComfyUI (flags never applied -> empty model lists, 2026-07-15)
    _comfy = subprocess.Popen(
        [sys.executable, os.path.join(COMFY_DIR, "main.py"),
         "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch",
         "--input-directory", INPUTS_DIR, "--output-directory", STAGE_DIR,
         "--extra-model-paths-config", EMP]
        # Cache policy. ComfyUI's default keeps EVERY node's output resident until
        # the prompt ends (built for interactive re-runs). For a one-shot pipeline
        # that is pure waste: on 2026-09-02 a 720-frame graph held every
        # intermediate batch at once and the container was OOM-killed at its
        # 125 GB limit. --cache-none frees a node's result once its consumers
        # have run (v0.33.1 flag: "Reduced RAM/VRAM usage at the expense of
        # executing every node for each run" — we run each graph once anyway).
        # PREP_COMFY_CACHE=classic restores the old behaviour for A/B.
        + ([] if os.environ.get("PREP_COMFY_CACHE", "none") == "classic" else ["--cache-none"]),
        cwd=COMFY_DIR, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if _comfy.poll() is not None:
            raise RuntimeError(f"ComfyUI exited rc={_comfy.returncode}.\n{_errs(LOG)}")
        try:
            # raw probe: a refused connection here is ComfyUI still starting,
            # not a death — no diagnosis, no log copies
            urllib.request.urlopen(URL + "/system_stats", timeout=5).read(); return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"ComfyUI boot timeout. log tail:\n{_tail(LOG)}")


def _node_watch(client_id, store):
    """Crash-guarded per-node telemetry: record when each node starts executing via the
    ComfyUI websocket. Never blocks or fails the job (partner principle)."""
    try:
        import websocket
        ws = websocket.create_connection(
            f"ws://127.0.0.1:{PORT}/ws?clientId={client_id}", timeout=10)
        ws.settimeout(5)
        store["ws"] = ws
        while not store.get("stop"):
            try:
                msg = ws.recv()
            except Exception:
                continue
            if not isinstance(msg, str):
                continue
            try:
                j = json.loads(msg)
            except ValueError:
                continue
            if j.get("type") == "executing":
                store["events"].append((time.time(), (j.get("data") or {}).get("node")))
    except Exception:
        pass


def _node_timings(events, graph, top_n=12):
    """events = [(t, node_id|None)] start markers -> per-node durations, labeled by class_type."""
    out = []
    for i, (t, nid) in enumerate(events):
        if nid is None:
            continue
        t_end = events[i + 1][0] if i + 1 < len(events) else t
        cls = (graph.get(str(nid)) or {}).get("class_type", "?")
        out.append({"node": str(nid), "class": cls, "s": round(t_end - t, 2)})
    out.sort(key=lambda x: -x["s"])
    return out[:top_n]


def _fetch_url(url, path, expected=None):
    """Resumable download to `path` (.part + Range retries). Long streams drop
    mid-transfer and a dropped connection reads as EOF (8/19: sam3.pt committed
    at 1.9 of 3.4 GB TWICE) — retry with Range from the .part offset until the
    size matches; never commit a byte count we can't verify. Returns (bytes,
    attempts); commits to `path` only when complete (or size unknown)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    attempts = 0
    got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    while attempts < 10:
        attempts += 1
        if expected is not None and got > expected:
            # Oversized partial = a previous resume appended a full-restart
            # response (server ignored the Range; observed 8/19 at exactly
            # +1 MiB). Corrupt by definition — restart clean.
            print(f"fetch: oversized partial {got}>{expected}, restarting", flush=True)
            os.remove(tmp)
            got = 0
        try:
            req = urllib.request.Request(url)
            if got:
                req.add_header("Range", f"bytes={got}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                status = getattr(r, "status", 200)
                if got and status != 206:
                    # Server ignored the Range — this response is the FULL
                    # file; appending it would corrupt. Restart from zero.
                    print(f"fetch: Range ignored (HTTP {status}), restarting", flush=True)
                    mode, got = "wb", 0
                else:
                    mode = "ab" if got else "wb"
                if expected is None and not got:
                    cl = r.headers.get("Content-Length")
                    if cl and status == 200:
                        expected = int(cl)
                with open(tmp, mode) as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise  # presigned URL expired/invalid — retrying cannot help
            print(f"fetch attempt {attempts} error at {got}B: HTTP {e.code}", flush=True)
        except Exception as e:  # noqa: BLE001 — transient network; retry from offset
            print(f"fetch attempt {attempts} error at {got}B: {type(e).__name__}: {e}", flush=True)
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expected is not None and got == expected:
            break
        if expected is None:
            break  # no size to verify against; single best-effort pass
        time.sleep(min(30, 3 * attempts))
    if expected is None or got == expected:
        os.replace(tmp, path)
    return got, attempts


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _put_url(url, path, attempts=3):
    """Upload a file to a presigned PUT URL (contract §4: 3 attempts, 2 s/4 s
    backoff). Returns the ETag. Raises on 401/403 immediately (expired URL —
    the backend must re-issue) and after the last failed attempt otherwise."""
    data = open(path, "rb").read()
    last = None
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, data=data, method="PUT",
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))})
            with urllib.request.urlopen(req, timeout=300) as r:
                return (r.headers.get("ETag") or "").strip('"')
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
        except Exception as e:  # noqa: BLE001 — transient network
            last = f"{type(e).__name__}: {e}"
        print(f"put attempt {i} failed: {last}", flush=True)
        if i < attempts:
            time.sleep(2 * i)
    raise RuntimeError(f"PUT failed after {attempts} attempts: {last}")


def _canonical_sha256(obj):
    """sha256 over byte-stable JSON — matches the backend's canonicalJson
    (sorted keys at every depth, no whitespace), so both sides can hash the
    same graph to the same hex and transport corruption of the settings is
    detectable (contract §1 echo)."""
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _find_newest(root, needle):
    best, best_m = None, -1
    for dirpath, _, files in os.walk(root):
        for f in files:
            if needle in f:
                p = os.path.join(dirpath, f)
                m = os.path.getmtime(p)
                if m > best_m: best, best_m = p, m
    return best


def handler(job):
    j = job.get("input") or {}
    if j.get("debug"):
        import glob as _glob
        hs_snaps = [s for g in prep_setup.HS_GLOBS for s in _glob.glob(g)]
        vol_models = f"{VOL}/prep-models/ComfyUI/models"
        vol_listing = {}
        for sub in ("unet", "loras", "text_encoders", "vae", "sam3"):
            p = os.path.join(vol_models, sub)
            vol_listing[sub] = sorted(os.listdir(p)) if os.path.isdir(p) else None
        # Live view while a warm-up is still running: ComfyUI's own log tail
        # (what it is loading / any error), network RX so far + the rate over
        # the last ~10 s (volume reads happen over the network), and the
        # warm-up nodes seen so far. Lets a ping mid-boot show the mechanism.
        pts = _NET["samples"]
        net_now = None
        if len(pts) >= 2:
            recent = [q for q in pts if q[0] >= pts[-1][0] - 10] or pts[-2:]
            rp = [(q[0], q[2]) for q in pts if len(q) > 2 and q[2] is not None]
            rrecent = [q for q in rp if q[0] >= rp[-1][0] - 10] if rp else []
            net_now = {"rx_gb_since_import": round((pts[-1][1] - pts[0][1]) / 2**30, 2),
                       "rate_mbs_last_10s": round((recent[-1][1] - recent[0][1]) / max(recent[-1][0] - recent[0][0], 1e-6) / 2**20, 1),
                       "comfy_read_gb": (round((rp[-1][1] - rp[0][1]) / 2**30, 2) if len(rp) >= 2 else None),
                       "comfy_read_mbs_last_10s": (round((rrecent[-1][1] - rrecent[0][1]) / max(rrecent[-1][0] - rrecent[0][0], 1e-6) / 2**20, 1) if len(rrecent) >= 2 else None),
                       "sampler_alive": not _NET["stop"], "age_s": round(time.time() - _T_IMPORT, 1)}
        return {"comfy_dir": COMFY_DIR, "inputs_dir": INPUTS_DIR, "gpu": _gpu_info(), "boot": BOOT,
                "net_now": net_now, "comfy_log_tail": _tail(LOG, 30), "comfy_alive": (_comfy is not None and _comfy.poll() is None),
                "warmup_events_so_far": [(round(t - _T_IMPORT, 1), nid) for t, nid in (_WARM_WATCH.get("events") or [])][-12:],
                "inputs_dir_exists": os.path.isdir(INPUTS_DIR),
                "stage_dir": STAGE_DIR, "emp_exists": os.path.isfile(EMP),
                # model visibility — the 8/19 failure needed console archaeology
                # to learn the hoststore never staged; now the ping says it.
                "hoststore_snapshots": hs_snaps,
                "hoststore_contents": {s: sorted(os.listdir(s))[:10] for s in hs_snaps},
                "volume_models": vol_listing,
                "inputs_sample": sorted(os.listdir(INPUTS_DIR))[:20] if os.path.isdir(INPUTS_DIR) else []}
    if j.get("iobench_par"):
        # Stream-count sweep: for each N, read `mb` MiB from a DIFFERENT region
        # of the file (so every pass hits uncached bytes) with N parallel
        # ranged readers. Finds the volume's parallelism knee on a warm worker.
        spec = j["iobench_par"]
        pth = os.path.realpath(spec.get("path") or os.path.join(PREP_MODELS_ROOT, "unet", "qwen-image-edit-2511-Q5_0.gguf"))
        limit = int(spec.get("mb", 1024)) * 2**20
        res = []
        off = int(spec.get("offset_mb", 0)) * 2**20
        for n in spec.get("streams", [1, 4, 8, 16, 32]):
            try:
                r = _read_ranged(pth, streams=int(n), limit_bytes=limit, offset=off)
                r["offset_gb"] = round(off / 2**30, 2)
                res.append(r)
            except Exception as e:
                res.append({"streams": n, "error": str(e)[:160]})
            off += limit
        return {"iobench_par": res, "gpu": _gpu_info(), "boot": BOOT}
    if j.get("find_big"):
        # Files >= min_gb under a root (depth-limited, time-capped): pick
        # uncached benchmark targets without guessing paths.
        spec = j["find_big"]
        root = spec.get("root", VOL); depth = int(spec.get("depth", 3)); min_b = float(spec.get("min_gb", 1)) * 2**30
        t0 = time.time(); hits = []
        for dp, dns, fns in os.walk(root):
            if dp[len(root):].count(os.sep) >= depth:
                dns[:] = []
            for fn in fns:
                fp = os.path.join(dp, fn)
                try:
                    sz = os.path.getsize(fp)
                    if sz >= min_b:
                        hits.append({"path": fp, "gb": round(sz / 2**30, 2)})
                except OSError:
                    pass
            if time.time() - t0 > float(spec.get("cap_s", 20)):
                break
        hits.sort(key=lambda h: -h["gb"])
        return {"find_big": hits[:40], "scan_s": round(time.time() - t0, 1)}
    if j.get("iobench"):
        # Sequential-read benchmark of one or more files (default: the SAM3
        # weights), N MiB each: answers "is the volume slow, or is lazy mmap
        # loading slow?" on a warm worker, no cold boot needed.
        spec = j["iobench"]
        paths = spec.get("paths") or [spec.get("path") or os.path.join(PREP_MODELS_ROOT, "sam3", "sam3.pt")]
        limit = int(spec.get("mb", 1024)) * 2**20
        res = []
        for pth in paths:
            try:
                res.append(_read_seq(os.path.realpath(pth), limit_bytes=limit))
            except Exception as e:
                res.append({"path": pth, "error": str(e)[:160]})
        return {"iobench": res, "gpu": _gpu_info(), "boot": BOOT}
    if j.get("fetch"):
        url, dest = j["fetch"]["url"], j["fetch"]["dest"]
        expected = j["fetch"].get("bytes")  # optional: enables resume + integrity
        if ".." in dest or dest.startswith("/"):
            return {"error": "dest must be a relative volume path"}
        path = os.path.join(VOL, dest)
        if os.path.exists(path) and not j["fetch"].get("overwrite"):
            return {"fetched": dest, "boot": BOOT, "gpu": _gpu_info(), "bytes": os.path.getsize(path), "skipped": "already exists"}
        t0 = time.time()
        try:
            got, attempts = _fetch_url(url, path, expected)
        except urllib.error.HTTPError as e:
            return {"error": f"fetch HTTP {e.code} (expired/invalid URL — re-issue and retry)", "fetched": dest}
        if expected is not None and got != expected:
            return {"error": f"fetch incomplete after {attempts} attempts: {got}/{expected} bytes",
                    "fetched": dest, "bytes": got}
        return {"fetched": dest, "boot": BOOT, "gpu": _gpu_info(), "bytes": got,
                "secs": round(time.time() - t0, 1), "attempts": attempts}
    if j.get("contract_version") is not None:
        return _contract_job(j)
    graph, outputs = j.get("graph"), j.get("outputs") or {}
    if not graph or not outputs:
        return {"error": "need input.graph and input.outputs"}
    t0 = time.time()
    _hwtele("phase", f"job_start_{int(t0)}")
    ensure_comfy()
    t_boot = time.time()

    _hwtele("phase", "graph_start")
    r = _run_graph(graph, int(j.get("timeout_s", 1500)), t0)
    if r.get("error"):
        _hwtele("phase", "graph_failed")
        return r["error"]
    stage, watch = r["stage"], r["watch"]
    t_graph = time.time()
    _hwtele("phase", "graph_end")

    written, missing = [], []
    for needle, final in outputs.items():
        src = _find_newest(stage, needle) or _find_newest(STAGE_DIR, needle)
        if not src:
            missing.append(needle); continue
        dst = os.path.join(INPUTS_DIR, final)
        shutil.move(src, dst)
        written.append({"file": final, "bytes": os.path.getsize(dst)})
    shutil.rmtree(stage, ignore_errors=True)
    if missing:
        return {"error": "missing outputs", "missing": missing,
                "written": written, "log_tail": _tail(LOG)}
    return {"written": written, "gpu": _gpu_info(), "boot": BOOT, "mem": _cgroup_mem(),
            "timing": {"boot_s": round(t_boot - t0, 1),
                       "graph_s": round(t_graph - t_boot, 1),
                       "total_s": round(time.time() - t0, 1)},
            "node_timings": _node_timings(watch["events"], graph)}


def _run_graph(graph, timeout_s, t0):
    """Submit + poll one graph in a fresh stage subdir (so needle matching
    can't hit stale files). Returns {stage, watch} on success or {"error":
    <ready-to-return dict>} on rejection/execution error/timeout. The watch
    thread is stopped either way."""
    stage = os.path.join(STAGE_DIR, f"job_{int(t0)}")
    for nid, node in graph.items():
        pref = node.get("inputs", {}).get("filename_prefix")
        if pref is not None:
            node["inputs"]["filename_prefix"] = f"job_{int(t0)}/" + pref

    client_id = f"prep_{int(t0)}"
    watch = {"events": [], "stop": False}
    threading.Thread(target=_node_watch, args=(client_id, watch), daemon=True).start()

    def _stop_watch():
        watch["stop"] = True
        try:
            if watch.get("ws"): watch["ws"].close()
        except Exception:
            pass

    try:
        pid = _http("/prompt", {"prompt": graph, "client_id": client_id}).get("prompt_id")
    except RuntimeError as e:
        _stop_watch()
        return {"error": {"error": "graph rejected", "detail": str(e)[:3000], "log_tail": _tail(LOG)}}
    if not pid:
        _stop_watch()
        return {"error": {"error": "submit failed", "log_tail": _tail(LOG)}}
    deadline = time.time() + timeout_s
    while True:
        try:
            h = _http(f"/history/{pid}")
        except RuntimeError as e:
            # ComfyUI stopped answering mid-graph (2026-09-02: two 720-frame
            # jobs, "[Errno 111] Connection refused", and the node they were
            # on was lost with the exception). Return the per-node record —
            # every node that STARTED, in order — so the failure names its
            # node, plus the diagnosis _http attached (process state, cgroup
            # memory + oom_kill, GPU memory, ComfyUI log).
            _stop_watch()
            ev = list(watch.get("events") or [])
            return {"error": {"error": "comfy unreachable mid-graph",
                              "detail": str(e)[:8000],
                              "nodes_started": _node_timings(ev, graph, top_n=64),
                              "last_events": [str(x)[:200] for x in ev[-5:]],
                              "log_tail": _tail(LOG)}}
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                break
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                _stop_watch()
                return {"error": {"error": "graph execution error",
                                  "detail": json.dumps(msgs)[:1500], "log_tail": _tail(LOG)}}
        if time.time() > deadline:
            _http("/interrupt", {})
            _stop_watch()
            return {"error": {"error": "graph timeout", "log_tail": _tail(LOG)}}
        time.sleep(5)
    _stop_watch()
    return {"stage": stage, "watch": watch}


def _contract_job(j):
    """Pipeline contract v1 (onset docs/design/pipeline-contract.md §3): ONE
    job fetches the inputs (byte+sha256-verified BEFORE any GPU work), runs the
    graph, PUTs each output to its presigned URL, and echoes what ran. Inputs
    land in INPUTS_DIR by bare name (local disk when the template points there;
    the volume until then). Transitional dual-write: outputs also land in
    INPUTS_DIR so the current render leg keeps reading them — dies with the
    render handler's fetch-by-URL (PREP_DUAL_WRITE=0)."""
    if j.get("contract_version") != 1:
        return {"error": "CONTRACT_MISMATCH", "detail": f"worker speaks contract 1, got {j.get('contract_version')!r}"}
    graph, outputs = j.get("graph"), j.get("outputs") or []
    if not graph or not outputs:
        return {"error": "need input.graph and input.outputs"}
    t0 = time.time()

    # 1. Inputs: fetch + verify BEFORE the GPU is touched (contract §4 — a bad
    #    transfer fails in seconds with no GPU spend).
    for spec in j.get("inputs") or []:
        name = spec["name"]
        if "/" in name or "\\" in name or ".." in name:
            return {"error": "FETCH_VERIFY_FAILED", "detail": f"input name {name!r} must be a bare filename"}
        path = os.path.join(INPUTS_DIR, name)
        try:
            got, attempts = _fetch_url(spec["url"], path, spec.get("bytes"))
        except urllib.error.HTTPError as e:
            return {"error": "URL_EXPIRED" if e.code in (401, 403) else "FETCH_VERIFY_FAILED",
                    "detail": f"input {name}: HTTP {e.code}"}
        if spec.get("bytes") is not None and got != spec["bytes"]:
            return {"error": "FETCH_VERIFY_FAILED",
                    "detail": f"input {name}: {got}/{spec['bytes']} bytes after {attempts} attempts"}
        if spec.get("sha256") and _sha256_file(path) != spec["sha256"]:
            return {"error": "FETCH_VERIFY_FAILED", "detail": f"input {name}: sha256 mismatch"}
    t_fetch = time.time()

    # 2. The graph. Hash it BEFORE _run_graph's stage-prefix rewrite mutates it —
    #    the echo must cover the settings exactly as the backend sent them.
    graph_hash = _canonical_sha256(graph)
    ensure_comfy()
    budget = int(j.get("budget_s", 1200))
    _hwtele("phase", f"job_start_{int(t0)}")
    _hwtele("phase", "graph_start")
    r = _run_graph(graph, budget, t0)
    if r.get("error"):
        _hwtele("phase", "graph_failed")
        return r["error"]
    _hwtele("phase", "graph_end")
    stage, watch = r["stage"], r["watch"]
    t_graph = time.time()

    # 3. Outputs: hash, PUT to storage, echo bytes+sha256+etag per artifact.
    dual = os.environ.get("PREP_DUAL_WRITE", "1") == "1"
    done, missing = [], []
    for spec in outputs:
        src = _find_newest(stage, spec["needle"]) or _find_newest(STAGE_DIR, spec["needle"])
        if not src:
            missing.append(spec["needle"]); continue
        entry = {"name": spec.get("name"), "key": spec.get("key"),
                 "bytes": os.path.getsize(src), "sha256": _sha256_file(src)}
        try:
            entry["etag"] = _put_url(spec["put_url"], src)
        except urllib.error.HTTPError as e:
            return {"error": "URL_EXPIRED" if e.code in (401, 403) else "PUT_FAILED",
                    "detail": f"output {spec['needle']}: HTTP {e.code}", "log_tail": _tail(LOG)}
        except Exception as e:  # noqa: BLE001
            return {"error": "PUT_FAILED", "detail": f"output {spec['needle']}: {str(e)[:300]}"}
        if dual and spec.get("name"):
            shutil.move(src, os.path.join(INPUTS_DIR, spec["name"]))
        done.append(entry)
    shutil.rmtree(stage, ignore_errors=True)
    if missing:
        return {"error": "missing outputs", "missing": missing, "outputs": done, "log_tail": _tail(LOG)}
    return {"mem": _cgroup_mem(),"contract_version": 1, "outputs": done,
            "params_effective": {"graph_sha256": graph_hash, "budget_s": budget},
            "timings": {"fetch_s": round(t_fetch - t0, 1), "run_s": round(t_graph - t_fetch, 1),
                        "put_s": round(time.time() - t_graph, 1)},
            "gpu": _gpu_info(), "boot": BOOT, "node_timings": _node_timings(watch["events"], graph)}


_WARM_WATCH = {"events": [], "stop": False}


def _boot_warmup():
    """Load every model at container start (SAM3 + Qwen GGUF/LoRA/CLIP/VAE via a
    tiny 1-step graph) so the first real job pays render time, not load time —
    the render endpoint's proven boot-warmup pattern. Crash-guarded: a warmup
    failure logs and serving proceeds; it must never take the worker down."""
    try:
        t0 = time.time()
        if os.environ.get("PREP_PREWARM", "1") == "1":
            # Page-cache pre-warm FIRST: sequential big-chunk reads of all model
            # files, so the lazy loads inside the graph hit RAM. Measured.
            try:
                BOOT["prewarm"] = _prewarm_models()
            except Exception as e:
                BOOT["prewarm"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            BOOT["prewarm_s"] = round(time.time() - t0, 1)
        else:
            BOOT["prewarm"] = "disabled"
        t_pw = time.time()
        ensure_comfy()
        t_comfy = time.time()
        BOOT["comfy_boot_s"] = round(t_comfy - t_pw, 1)
        with open("/opt/prep_warmup_graph.json") as f:
            g = json.load(f)
        # Per-node timings of the warmup graph = per-model LOAD times (SAM3,
        # Qwen GGUF, CLIP, VAE, ...) — the numbers that say where a cold boot
        # spends its time and whether the model store is doing its job.
        watch = _WARM_WATCH
        threading.Thread(target=_node_watch, args=("boot_warmup", watch), daemon=True).start()
        pid = _http("/prompt", {"prompt": g, "client_id": "boot_warmup"}).get("prompt_id")
        # Volume loads measured 8/21 at ~7 min on L40S; the wait must outlast
        # them or the timeline is truncated (the graph keeps running regardless).
        deadline = time.time() + 1200
        outcome = "timeout"
        while pid and time.time() < deadline:
            h = _http(f"/history/{pid}")
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("completed") or st.get("status_str") in ("success", "error"):
                    outcome = "ok" if st.get("status_str") != "error" else "error"
                    if outcome == "error":
                        print(f"BOOT_WARMUP graph error: {json.dumps(st)[:800]}", flush=True)
                        BOOT["warmup_error"] = json.dumps(st)[:600]
                    break
            time.sleep(2)
        watch["stop"] = True
        try:
            if watch.get("ws"): watch["ws"].close()
        except Exception:
            pass
        t_end = time.time()
        BOOT["warmup_s"] = round(t_end - t_comfy, 1)
        BOOT["warmup"] = outcome
        nodes = _node_timings(watch["events"], g, top_n=8)
        # network bytes per node window = what each load actually pulled from the volume
        ev = [(t, nid) for t, nid in watch["events"] if nid is not None]
        for n in nodes:
            for i, (t, nid) in enumerate(ev):
                if str(nid) == n["node"]:
                    t_next = ev[i + 1][0] if i + 1 < len(ev) else t_end
                    n["net"] = _net_window(t, t_next)
                    break
        BOOT["warmup_nodes"] = nodes
        BOOT["warmup_net"] = _net_window(t_comfy, t_end)
        _NET["stop"] = True
        BOOT["ready_after_import_s"] = round(time.time() - _T_IMPORT, 1)
        print(f"BOOT_WARMUP done in {time.time() - t0:.1f}s {json.dumps(BOOT)}", flush=True)
    except Exception as e:
        BOOT["warmup"] = f"skipped: {type(e).__name__}: {str(e)[:120]}"
        BOOT["ready_after_import_s"] = round(time.time() - _T_IMPORT, 1)
        print(f"BOOT_WARMUP skipped: {type(e).__name__}: {e}", flush=True)


if os.environ.get("PREP_BOOT_WARMUP") == "1":
    # NON-BLOCKING (8/21): serving starts immediately. A cold worker's first
    # job is the backend's clip fetch (no ComfyUI) - it returns in seconds
    # instead of waiting ~7 min behind the loads; the graph job then queues
    # behind the warm-up prompt inside ComfyUI (sequential), so it starts
    # exactly when the models are resident. Total latency unchanged, budgets
    # and progress reporting honest.
    threading.Thread(target=_boot_warmup, daemon=True).start()
else:
    BOOT["warmup"] = "disabled"
    BOOT["ready_after_import_s"] = round(time.time() - _T_IMPORT, 1)
    _NET["stop"] = True

runpod.serverless.start({"handler": handler})
