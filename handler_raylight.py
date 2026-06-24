#!/usr/bin/env python3
"""RunPod Serverless handler — Raylight (ComfyUI + Wan2.1-VACE-14B Q4 GGUF + FusionX) USP render.

NET-NEW path (2026-06-24). The native bf16 75GB serverless path worked but was closed on cost/fit; this wraps the
Raylight GGUF stack (proven on pods: VACE survives USP at 1/2/4/8) as a serverless worker, to characterize the
cold-load / warm-worker economics a pod cannot show. **Starting at 1 GPU** (cheapest, simplest, answers the crux
without USP); n_gpus>1 supported for later.

ARCHITECTURE (persistent_server — the warm path must be architecturally possible, then VERIFIED at runtime):
  - The runpod PARENT process stays OFF the GPU (native fix: a ~730MB parent CUDA ctx tipped OOM). ComfyUI runs as a
    SUBPROCESS that owns the GPU(s); Ray actors are its children.
  - ComfyUI+Raylight is launched ONCE at module-load and kept alive => a WARM worker keeps the dequantized model
    resident. handler(job) only submits the workflow to local ComfyUI :8188 and polls /history.
  - Env+ComfyUI+nodes baked in the image (COMFY_DIR); the ~18GB model set + inputs on the mounted volume.

CRUX LEVER: RayInitializer.clear_vram_after_sampling forced FALSE so the model stays resident across requests.
Whether that TRULY holds (vs silent ~300s re-dequant, or RunPod killing the worker at idleTimeout=5s first) is what
the telemetry measures: cold/ordinal/epoch + VRAM-residency-at-entry + total time → the consensus class label.

TELEMETRY (per cloud_benchmark/serverless_monitoring_spec.md): worker self-narrates to the VOLUME via append+fsync
beacons (boot / req_entry / req_exit / heartbeat / death) because the worker EVAPORATES at idleTimeout taking its
container FS with it. Assume SIGKILL → per-request flush is the safety net, the death beacon is a bonus. The FULL
in-worker RENDER telemetry (§1-§2c: DCGM/NVLink/NCCL/py-spy/nsys/quality) plugs into _render_telemetry_start/stop.
Precise RayGGUFLoader dequant-duration needs wrapping the Raylight loader node (deeper hook, flagged in the spec);
here the VRAM-residency probe + total time give the practical crux signal.

UNKNOWNS verified on first invoke (flagged, NOT assumed): ComfyUI+Ray persisting across warm invokes; clear_vram=False
keeping the model resident; the exact on-volume model/input paths; whether RunPod sends SIGTERM(grace) or SIGKILL.
"""
import os, sys, time, json, glob, base64, signal, atexit, subprocess, threading, urllib.request, urllib.error

# Parent OFF the GPU; remember what RunPod assigned so we hand it to the ComfyUI subprocess only.
_NVIS = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import runpod

VOL        = os.environ.get("VOL", "/runpod-volume")
COMFY_DIR  = os.environ.get("COMFY_DIR", "/opt/ComfyUI")
HERE       = os.path.dirname(os.path.abspath(__file__))
WF_PATH    = os.environ.get("WF_PATH", os.path.join(HERE, "raylight_vace_wf.json"))
INPUTS_DIR = os.environ.get("INPUTS_DIR", f"{VOL}/native-xdit/inputs")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/slout")
ENDPOINT   = os.environ.get("RUNPOD_ENDPOINT_ID", "sl")
PORT       = int(os.environ.get("COMFY_PORT", "8188"))
URL        = f"http://127.0.0.1:{PORT}"
WORKER_ID  = os.environ.get("RUNPOD_POD_ID") or os.environ.get("HOSTNAME") or f"pid{os.getpid()}"

_BOOT_T = time.time()
# module_load_epoch: unique per worker PROCESS — SAME across requests on a warm worker, NEW on every cold boot.
MODULE_EPOCH = f"{WORKER_ID}.{int(_BOOT_T)}.{os.getpid()}"
WDIR = os.path.join(os.environ.get("TELE_DIR", f"{VOL}/serverless_telemetry"), ENDPOINT, WORKER_ID)

_state = {"ready_t": None, "renders": 0, "comfy_pid": None}
_comfy = None


# --- low-level helpers -----------------------------------------------------------------------------------------
def _mk(d):
    try: os.makedirs(d, exist_ok=True)
    except Exception: pass


def _run(cmd, timeout=20):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as e: return f"ERR {e}"


def _n_gpus():
    out = _run(["nvidia-smi", "-L"])
    return max(1, sum(1 for l in out.splitlines() if l.strip().startswith("GPU ")))


def _vram_used():
    """memory.used MiB per GPU — the VRAM-residency probe (warm worker should show the model still resident)."""
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    vals = []
    for ln in out.splitlines():
        try: vals.append(int(ln.strip()))
        except Exception: pass
    return vals


def _get(path, timeout=15):
    return json.loads(urllib.request.urlopen(URL + path, timeout=timeout).read().decode())


def _beacon(kind, **rec):
    """Append+fsync one JSONL line to the volume. Survives SIGKILL — the per-request/lifecycle safety net."""
    _mk(WDIR)
    line = {"kind": kind, "t": time.time(), "mono": time.monotonic(),
            "endpoint": ENDPOINT, "worker_id": WORKER_ID, "epoch": MODULE_EPOCH, "ordinal": _state["renders"], **rec}
    try:
        with open(os.path.join(WDIR, "beacon.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n"); f.flush(); os.fsync(f.fileno())
    except Exception:
        pass


# --- boot record: identity + dirty-GPU (prior-tenant residue, BEFORE we load anything) + arch fingerprint ------
def _write_boot_record():
    _mk(WDIR)
    rec = {
        "schema": 1, "endpoint": ENDPOINT, "worker_id": WORKER_ID, "epoch": MODULE_EPOCH,
        "boot_wall": _BOOT_T, "pid": os.getpid(), "n_gpus": _n_gpus(),
        "gpus": _run(["nvidia-smi", "--query-gpu=name,uuid,pcie.link.gen.max,memory.total",
                      "--format=csv,noheader"]),
        "nvlink": _run(["nvidia-smi", "nvlink", "-s"]),
        "dirty_gpu_vram_used_mib": _vram_used(),                 # prior-tenant residue → OOM-by-150MB risk
        "compute_apps_at_boot": _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]),
        "cuda_visible_assigned": _NVIS,
        "handler_arch": "persistent_server",                    # render runs IN the long-lived ComfyUI (no per-req subprocess)
        "image_digest": os.environ.get("RUNPOD_IMAGE_NAME", ""),
        "config_env": {k: v for k, v in os.environ.items()
                       if k.startswith(("RUNPOD_", "MODEL_", "WORKERS", "IDLE", "FLASH", "CLEAR_VRAM"))},
    }
    try: json.dump(rec, open(os.path.join(WDIR, "identity.json"), "w"), indent=2)
    except Exception: pass
    _beacon("boot", dirty_vram=rec["dirty_gpu_vram_used_mib"], n_gpus=rec["n_gpus"])


def _heartbeat():
    """Idle heartbeat: disambiguates worker death-vs-alive and tracks VRAM residency across idle gaps."""
    while True:
        time.sleep(15)
        _beacon("heartbeat", vram=_vram_used(), comfy_alive=(_comfy.poll() is None if _comfy else False))


def _on_signal(sig, frame):
    # SIGTERM marker present in the beacon => RunPod gave a grace window; absent => SIGKILL (per-req flush is the net).
    _beacon("death", trigger="SIGTERM", signal=int(sig))
    os._exit(0)


# --- ComfyUI lifecycle -----------------------------------------------------------------------------------------
def _launch_comfy():
    global _comfy
    _mk(WDIR); _mk(OUTPUT_DIR)
    n = _n_gpus()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = _NVIS if _NVIS else ",".join(str(i) for i in range(n))   # GPUs to ComfyUI/Ray
    env["COMFY_DIR"] = COMFY_DIR
    # Baked ComfyUI finds the ~18GB model set on the mounted VOLUME via an extra-model-paths config (models not baked).
    cargs = f"--input-directory {INPUTS_DIR} --output-directory {OUTPUT_DIR}"
    emp = os.environ.get("EXTRA_MODEL_PATHS", "/opt/extra_model_paths.yaml")
    if emp and os.path.exists(emp):
        cargs += f" --extra-model-paths-config {emp}"
    env["COMFY_ARGS"] = cargs
    env["PYTHONUNBUFFERED"] = "1"
    boot_log = open(os.path.join(WDIR, "comfy.log"), "ab", buffering=0)
    _beacon("comfy_launch")
    _comfy = subprocess.Popen([sys.executable, os.path.join(HERE, "comfy_launch.py")],
                              env=env, stdout=boot_log, stderr=subprocess.STDOUT, cwd=COMFY_DIR)
    _state["comfy_pid"] = _comfy.pid
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            _get("/system_stats", timeout=5)
            _state["ready_t"] = time.time()
            _beacon("comfy_ready", boot_to_ready_s=round(_state["ready_t"] - _BOOT_T, 1), comfy_pid=_comfy.pid)
            return
        except Exception:
            if _comfy.poll() is not None:
                _beacon("comfy_exit_during_boot", rc=_comfy.returncode)
                raise RuntimeError(f"ComfyUI exited during boot (rc={_comfy.returncode}) — see {boot_log.name}")
            time.sleep(2)
    raise RuntimeError("ComfyUI did not become ready within 900s")


def _build_wf(job, n):
    wf = json.load(open(WF_PATH))
    length = int(job.get("frame_num", 81))
    steps  = int(job.get("sample_steps", 6))
    w, h   = int(job.get("width", 720)), int(job.get("height", 1280))
    wf["1"]["inputs"]["GPU"] = n
    wf["1"]["inputs"]["ulysses_degree"] = n
    wf["1"]["inputs"]["clear_vram_after_sampling"] = False   # WARM LEVER — keep model resident across requests
    for node, key in (("9", "src_video"), ("10", "src_mask")):
        if job.get(key):
            wf[node]["inputs"]["video"] = os.path.basename(job[key])
        wf[node]["inputs"]["custom_width"], wf[node]["inputs"]["custom_height"] = w, h
        wf[node]["inputs"]["frame_load_cap"] = length
    if job.get("src_ref_images"):
        wf["12"]["inputs"]["image"] = os.path.basename(job["src_ref_images"])
    for nd in ("13", "14"):
        wf[nd]["inputs"]["width"], wf[nd]["inputs"]["height"] = w, h
    wf["14"]["inputs"]["length"] = length
    wf["15"]["inputs"]["steps"] = steps
    if job.get("prompt"):
        wf["7"]["inputs"]["text"] = job["prompt"]
    wf["18"]["inputs"]["filename_prefix"] = f"SLBENCH/{WORKER_ID}_{int(time.time())}"
    return wf, {"length": length, "steps": steps, "width": w, "height": h, "n_gpus": n}


# --- render telemetry hook (v1 light 1 Hz curve; FULL §1-§2c render capture wires in HERE from the spec) -------
def _render_telemetry_start(reqdir):
    stop = threading.Event()
    def sample():
        try:
            f = open(os.path.join(reqdir, "gpu_1hz.csv"), "w", buffering=1)
        except Exception:
            return
        f.write("t,gpu_util,vram_mib,power_w\n")
        q = "nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits".split()
        while not stop.is_set():
            lines = _run(q).splitlines()
            f.write(f"{time.time():.1f},{lines[0].replace(' ', '') if lines else ''}\n")
            stop.wait(1.0)
        f.close()
    threading.Thread(target=sample, daemon=True).start()
    return stop


def _render_telemetry_stop(stop):
    if stop: stop.set()
    time.sleep(0.2)   # let the last sample flush to the volume


def handler(event):
    job = (event or {}).get("input", {}) or {}
    if job.get("debug"):
        return {"worker_id": WORKER_ID, "epoch": MODULE_EPOCH, "handler_arch": "persistent_server",
                "n_gpus": _n_gpus(), "comfy_ready": _state["ready_t"] is not None,
                "renders_on_this_worker": _state["renders"], "vram_now": _vram_used(),
                "boot_to_ready_s": (_state["ready_t"] or _BOOT_T) - _BOOT_T}
    if not job.get("prompt"):
        return {"error": "prompt is required"}

    cold = (_state["renders"] == 0)                     # cold = first render on this worker (pays the dequant)
    n = int(job.get("n_gpus") or _n_gpus())
    jid = (event or {}).get("id") or f"{WORKER_ID}_{_state['renders']}"
    vram_entry = _vram_used()
    resident = bool(vram_entry and max(vram_entry) > 8000)   # >8GB => dequantized GGUF likely resident (warm)
    t_entry = time.time()
    _beacon("req_entry", job_id=jid, cold=cold, n_gpus=n, vram_entry=vram_entry,
            model_resident_at_entry=resident, worker_age_s=round(t_entry - _BOOT_T, 1))

    reqdir = os.path.join(WDIR, str(jid)); _mk(reqdir)
    wf, meta = _build_wf(job, n)
    stop = _render_telemetry_start(reqdir)
    t_submit = time.time()
    try:
        body = json.dumps({"prompt": wf, "client_id": str(jid)}).encode()
        req = urllib.request.Request(URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
        try:
            pid = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["prompt_id"]
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:1500]
            _beacon("req_error", job_id=jid, stage="submit", detail=err[:300])
            return {"error": "workflow validation failed", "detail": err, "cold": cold, "worker_id": WORKER_ID}

        err = None
        while time.time() - t_submit < 2400:
            time.sleep(3)
            try: h = _get(f"/history/{pid}")
            except Exception: continue
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") != "success":
                    err = json.dumps(st.get("messages", []))[:1500]
                break
        t_done = time.time()
        vram_exit = _vram_used()

        mp4s = sorted(glob.glob(os.path.join(OUTPUT_DIR, wf["18"]["inputs"]["filename_prefix"] + "*.mp4")))
        out = {
            "worker_id": WORKER_ID, "epoch": MODULE_EPOCH, "job_id": jid, "ordinal": _state["renders"],
            "cold": cold, "model_resident_at_entry": resident,                 # consensus inputs for the class label
            "worker_age_s": round(t_entry - _BOOT_T, 1),
            "boot_to_ready_s": round((_state["ready_t"] or _BOOT_T) - _BOOT_T, 1),
            "total_s": round(t_done - t_submit, 1),       # cold ~= dequant + sample; warm-resident ~= sample only
            "vram_entry_mib": vram_entry, "vram_exit_mib": vram_exit, **meta,
        }
        if err:
            out["error"] = "render failed"; out["detail"] = err
        elif mp4s:
            data = open(mp4s[-1], "rb").read()
            out["bytes"] = len(data)
            # TODO(long clips): switch to an S3 URL once outputs exceed the response ceiling (spec G).
            out["video_base64"] = base64.b64encode(data).decode()
        else:
            out["error"] = "no output produced"

        # durable per-request record: write *.partial then atomic os.replace (survives an idle-out mid-write)
        rec = {k: v for k, v in out.items() if k != "video_base64"}
        tmp = os.path.join(reqdir, "request.json.partial")
        try:
            json.dump(rec, open(tmp, "w"), indent=2); os.replace(tmp, os.path.join(reqdir, "request.json"))
        except Exception:
            pass
        _beacon("req_exit", job_id=jid, total_s=out["total_s"], ok=("error" not in out),
                bytes=out.get("bytes"), vram_exit=vram_exit, model_resident_at_entry=resident)
        return out
    finally:
        _render_telemetry_stop(stop)
        _state["renders"] += 1


# --- module load: capture dirty-GPU/identity FIRST, then launch the persistent worker, then serve --------------
atexit.register(lambda: _beacon("death", trigger="atexit"))
try: signal.signal(signal.SIGTERM, _on_signal)
except Exception: pass

_write_boot_record()                 # BEFORE ComfyUI touches the GPU (captures prior-tenant dirty VRAM)
threading.Thread(target=_heartbeat, daemon=True).start()
_launch_comfy()                      # persistent ComfyUI+Ray so a warm worker keeps the model resident
runpod.serverless.start({"handler": handler})
