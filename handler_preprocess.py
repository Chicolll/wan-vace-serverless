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
def _net_sampler():
    while not _NET["stop"]:
        try:
            _NET["samples"].append((time.time(), _net_rx_bytes()))
        except Exception:
            pass
        time.sleep(1)
threading.Thread(target=_net_sampler, daemon=True).start()
def _net_window(t_a, t_b):
    """RX bytes received in [t_a, t_b] and the peak 1 s rate (MB/s) inside it."""
    pts = [(t, b) for t, b in _NET["samples"] if t_a - 1 <= t <= t_b + 1]
    if len(pts) < 2:
        return {"gb": None, "peak_mbs": None}
    peak = max((b2 - b1) / max(t2 - t1, 1e-6) for (t1, b1), (t2, b2) in zip(pts, pts[1:]))
    return {"gb": round((pts[-1][1] - pts[0][1]) / 2**30, 2), "peak_mbs": round(peak / 2**20)}
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
            net_now = {"rx_gb_since_import": round((pts[-1][1] - pts[0][1]) / 2**30, 2),
                       "rate_mbs_last_10s": round((recent[-1][1] - recent[0][1]) / max(recent[-1][0] - recent[0][0], 1e-6) / 2**20, 1),
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
        ensure_comfy()
        t_comfy = time.time()
        BOOT["comfy_boot_s"] = round(t_comfy - t0, 1)
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
