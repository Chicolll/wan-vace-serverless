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
    files = sorted(set(files), key=lambda x: -os.path.getsize(x))
    streams = int(threads or os.environ.get("PREP_PREWARM_STREAMS", "16"))
    out = {"files": [], "n": len(files), "streams": streams}
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


def _http(path, payload=None, timeout=30):
    req = urllib.request.Request(URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:3000]
        raise RuntimeError(f"ComfyUI {path} HTTP {e.code}: {body}")


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
         "--extra-model-paths-config", EMP],
        cwd=COMFY_DIR, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if _comfy.poll() is not None:
            raise RuntimeError(f"ComfyUI exited rc={_comfy.returncode}.\n{_errs(LOG)}")
        try:
            _http("/system_stats"); return
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
        hs_snaps = _glob.glob("/runpod/model-store/huggingface/Chicolll/bg-replace-pipeline/*/snapshots/*/")
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.time()
        tmp = path + ".part"
        # Resumable download: long streams drop mid-transfer and a dropped
        # connection reads as EOF (8/19: sam3.pt committed at 1.9 of 3.4 GB
        # TWICE) — so retry with Range from the .part offset until the size
        # matches, and never commit a byte count we can't verify.
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
            except Exception as e:  # noqa: BLE001 — transient network; retry from offset
                print(f"fetch attempt {attempts} error at {got}B: {type(e).__name__}: {e}", flush=True)
            got = os.path.getsize(tmp)
            if expected is not None and got == expected:
                break
            if expected is None:
                break  # no size to verify against; single best-effort pass
            time.sleep(min(30, 3 * attempts))
        if expected is not None and got != expected:
            return {"error": f"fetch incomplete after {attempts} attempts: {got}/{expected} bytes",
                    "fetched": dest, "bytes": got}
        os.replace(tmp, path)
        return {"fetched": dest, "boot": BOOT, "gpu": _gpu_info(), "bytes": os.path.getsize(path),
                "secs": round(time.time() - t0, 1), "attempts": attempts}
    graph, outputs = j.get("graph"), j.get("outputs") or {}
    if not graph or not outputs:
        return {"error": "need input.graph and input.outputs"}
    t0 = time.time()
    ensure_comfy()
    t_boot = time.time()

    # fresh stage subdir per job so needle matching can't hit stale files
    stage = os.path.join(STAGE_DIR, f"job_{int(t0)}")
    for nid, node in graph.items():
        pref = node.get("inputs", {}).get("filename_prefix")
        if pref is not None:
            node["inputs"]["filename_prefix"] = f"job_{int(t0)}/" + pref

    import threading
    client_id = f"prep_{int(t0)}"
    watch = {"events": [], "stop": False}
    threading.Thread(target=_node_watch, args=(client_id, watch), daemon=True).start()
    try:
        pid = _http("/prompt", {"prompt": graph, "client_id": client_id}).get("prompt_id")
    except RuntimeError as e:
        watch["stop"] = True
        return {"error": "graph rejected", "detail": str(e)[:3000], "log_tail": _tail(LOG)}
    if not pid:
        watch["stop"] = True
        return {"error": "submit failed", "log_tail": _tail(LOG)}
    deadline = time.time() + int(j.get("timeout_s", 1500))
    while True:
        h = _http(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                break
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                return {"error": "graph execution error",
                        "detail": json.dumps(msgs)[:1500], "log_tail": _tail(LOG)}
        if time.time() > deadline:
            _http("/interrupt", {})
            return {"error": "graph timeout", "log_tail": _tail(LOG)}
        time.sleep(5)
    t_graph = time.time()

    written, missing = [], []
    for needle, final in outputs.items():
        src = _find_newest(stage, needle) or _find_newest(STAGE_DIR, needle)
        if not src:
            missing.append(needle); continue
        dst = os.path.join(INPUTS_DIR, final)
        shutil.move(src, dst)
        written.append({"file": final, "bytes": os.path.getsize(dst)})
    shutil.rmtree(stage, ignore_errors=True)
    watch["stop"] = True
    try:
        if watch.get("ws"): watch["ws"].close()
    except Exception:
        pass
    if missing:
        return {"error": "missing outputs", "missing": missing,
                "written": written, "log_tail": _tail(LOG)}
    return {"written": written, "gpu": _gpu_info(), "boot": BOOT,
            "timing": {"boot_s": round(t_boot - t0, 1),
                       "graph_s": round(t_graph - t_boot, 1),
                       "total_s": round(time.time() - t0, 1)},
            "node_timings": _node_timings(watch["events"], graph)}


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
