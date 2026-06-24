#!/usr/bin/env python3
"""RunPod Serverless handler — Raylight (ComfyUI + Wan2.1-VACE-14B Q4 GGUF + FusionX) USP render.

NET-NEW path. Robust serverless design (2026-06-24, after a first deploy where workers exited unhealthy because
module-load blocked on ComfyUI before runpod.serverless.start() registered):
  - runpod.serverless.start() is called IMMEDIATELY at module-load so the worker registers healthy fast.
  - ComfyUI+Raylight is launched in a BACKGROUND thread (non-blocking) and warms while the worker is healthy.
  - EVERY file op is crash-guarded; telemetry falls back to /tmp if the volume isn't writable; key events print to
    stdout (RunPod logs). handler(debug) returns full diagnostics (volume / models / env / GPU / comfy status)
    WITHOUT needing ComfyUI, so failures are visible from a cheap debug invoke instead of an opaque worker exit.
  - Parent process stays OFF the GPU (native fix: a ~730MB parent CUDA ctx tipped OOM); GPUs go to the ComfyUI
    subprocess + Ray actors. clear_vram_after_sampling=False (the warm-residency lever).
"""
import os, sys, time, json, glob, base64, traceback, subprocess, threading, urllib.request, urllib.error


def log(*a):
    try: print("[handler]", *a, flush=True)
    except Exception: pass


# Parent OFF the GPU; remember RunPod's assignment for the ComfyUI subprocess.
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
_BOOT_T    = time.time()
MODULE_EPOCH = f"{WORKER_ID}.{int(_BOOT_T)}.{os.getpid()}"


def _pick_tele_dir():
    """Telemetry dir on the VOLUME if writable, else /tmp. Returns (dir, on_volume)."""
    base = os.environ.get("TELE_DIR", f"{VOL}/serverless_telemetry")
    cand = os.path.join(base, ENDPOINT, WORKER_ID)
    try:
        os.makedirs(cand, exist_ok=True)
        p = os.path.join(cand, ".writetest"); open(p, "w").close(); os.remove(p)
        return cand, True
    except Exception as e:
        log("VOLUME telemetry dir NOT writable:", repr(e))
        alt = os.path.join("/tmp/sltele", ENDPOINT, WORKER_ID)
        try: os.makedirs(alt, exist_ok=True)
        except Exception: pass
        return alt, False


WDIR, VOL_WRITABLE = _pick_tele_dir()
_comfy = None
_comfy_state = {"phase": "not_started", "error": None, "ready": False, "pid": None, "started_t": None}
_comfy_lock = threading.Lock()


def _run(cmd, timeout=20):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as e: return f"ERR {e}"


def _n_gpus():
    out = _run(["nvidia-smi", "-L"])
    return max(1, sum(1 for l in out.splitlines() if l.strip().startswith("GPU ")))


def _vram_used():
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = []
    for ln in out.splitlines():
        try: v.append(int(ln.strip()))
        except Exception: pass
    return v


def _get(path, timeout=10):
    return json.loads(urllib.request.urlopen(URL + path, timeout=timeout).read().decode())


def _tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - n))
            return f.read().decode("utf-8", "replace")
    except Exception as e:
        return f"(no log: {e})"


def _beacon(kind, **rec):
    try:
        line = {"kind": kind, "t": time.time(), "worker_id": WORKER_ID, "epoch": MODULE_EPOCH, **rec}
        with open(os.path.join(WDIR, "beacon.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n"); f.flush(); os.fsync(f.fileno())
    except Exception:
        pass


def _ensure_comfy(timeout=900):
    """Launch ComfyUI+Raylight ONCE (idempotent). NON-fatal: any error is recorded in _comfy_state, never raised."""
    global _comfy
    with _comfy_lock:
        if _comfy_state["ready"]:
            return True
        if _comfy_state["phase"] in ("launching", "error", "timeout", "exception"):
            # another caller is launching, or it already failed; just wait/return current state
            pass
        if _comfy is None and _comfy_state["phase"] in ("not_started", "launching"):
            _comfy_state["phase"] = "launching"; _comfy_state["started_t"] = time.time()
            start_now = True
        else:
            start_now = False
    if start_now:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            n = _n_gpus()
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = _NVIS if _NVIS else ",".join(str(i) for i in range(n))
            env["COMFY_DIR"] = COMFY_DIR
            cargs = f"--input-directory {INPUTS_DIR} --output-directory {OUTPUT_DIR}"
            emp = os.environ.get("EXTRA_MODEL_PATHS", "/opt/extra_model_paths.yaml")
            if emp and os.path.exists(emp):
                cargs += f" --extra-model-paths-config {emp}"
            env["COMFY_ARGS"] = cargs
            env["PYTHONUNBUFFERED"] = "1"
            logp = os.path.join(WDIR, "comfy.log")
            try: out_f = open(logp, "ab", buffering=0)
            except Exception: out_f = None
            log(f"launching ComfyUI: COMFY_DIR={COMFY_DIR} args={cargs} gpus={env['CUDA_VISIBLE_DEVICES']}")
            _comfy = subprocess.Popen([sys.executable, os.path.join(HERE, "comfy_launch.py")],
                                      env=env, cwd=COMFY_DIR,
                                      stdout=(out_f or subprocess.DEVNULL), stderr=subprocess.STDOUT)
            _comfy_state["pid"] = _comfy.pid
            _beacon("comfy_launch", pid=_comfy.pid)
        except Exception:
            _comfy_state["phase"] = "exception"; _comfy_state["error"] = traceback.format_exc()
            log("ensure_comfy launch EXCEPTION:", _comfy_state["error"]); return False
    # wait for readiness (any caller)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _comfy_state["ready"]:
            return True
        try:
            _get("/system_stats", timeout=5)
            _comfy_state["ready"] = True; _comfy_state["phase"] = "ready"
            log(f"ComfyUI READY in {round(time.time()-(_comfy_state['started_t'] or t0),1)}s")
            _beacon("comfy_ready", boot_to_ready_s=round(time.time() - _BOOT_T, 1))
            return True
        except Exception:
            if _comfy is not None and _comfy.poll() is not None:
                _comfy_state["phase"] = "error"
                _comfy_state["error"] = f"ComfyUI exited rc={_comfy.returncode}. log tail:\n{_tail(os.path.join(WDIR,'comfy.log'))}"
                log("ComfyUI EXITED:", _comfy_state["error"]); _beacon("comfy_exit", rc=_comfy.returncode)
                return False
            time.sleep(2)
    _comfy_state["phase"] = "timeout"; _comfy_state["error"] = "ComfyUI not ready within timeout"
    return False


def _debug_models():
    base = os.path.join(VOL, "runpod-slim", "ComfyUI", "models")
    want = {
        "gguf":     os.path.join(base, "unet", "gguf", "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf"),
        "lora":     os.path.join(base, "loras", "Wan2.1_T2V_14B_FusionX_LoRA.safetensors"),
        "clip":     os.path.join(base, "clip", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        "clip_alt": os.path.join(base, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        "vae":      os.path.join(base, "vae", "wan_2.1_vae.safetensors"),
    }
    out = {}
    for k, p in want.items():
        try:
            lex = os.path.lexists(p); real = os.path.realpath(p) if lex else None
            out[k] = {"path": p, "exists": os.path.exists(p), "is_symlink": os.path.islink(p) if lex else False,
                      "realpath": real, "real_exists": (os.path.exists(real) if real else False),
                      "size": (os.path.getsize(p) if os.path.exists(p) else None)}
        except Exception as e:
            out[k] = {"path": p, "err": repr(e)}
    return out


def _debug():
    """Full diagnostics — does NOT need ComfyUI. Surfaces the exact reason a render would fail."""
    d = {
        "worker_id": WORKER_ID, "epoch": MODULE_EPOCH, "boot_to_now_s": round(time.time() - _BOOT_T, 1),
        "handler_arch": "persistent_server_lazy", "n_gpus": _n_gpus(), "vram_used_mib": _vram_used(),
        "telemetry_dir": WDIR, "telemetry_on_volume": VOL_WRITABLE,
        "comfy": dict(_comfy_state),
        "env_runpod": {k: v for k, v in os.environ.items() if k.startswith(("RUNPOD_", "MODEL_"))},
        "paths": {"VOL": VOL, "vol_exists": os.path.isdir(VOL),
                  "comfy_dir": COMFY_DIR, "comfy_dir_exists": os.path.isdir(COMFY_DIR),
                  "inputs_dir": INPUTS_DIR, "inputs_dir_exists": os.path.isdir(INPUTS_DIR),
                  "wf_path": WF_PATH, "wf_exists": os.path.exists(WF_PATH)},
    }
    try: d["vol_listing"] = sorted(os.listdir(VOL))[:25]
    except Exception as e: d["vol_listing"] = f"ERR {e}"
    d["models"] = _debug_models()
    if _comfy_state.get("phase") in ("error", "exception", "timeout"):
        d["comfy_log_tail"] = _tail(os.path.join(WDIR, "comfy.log"))
    return d


def _build_wf(job, n):
    wf = json.load(open(WF_PATH))
    length = int(job.get("frame_num", 81)); steps = int(job.get("sample_steps", 6))
    w, h = int(job.get("width", 720)), int(job.get("height", 1280))
    wf["1"]["inputs"]["GPU"] = n
    wf["1"]["inputs"]["ulysses_degree"] = n
    wf["1"]["inputs"]["clear_vram_after_sampling"] = False
    for node, key in (("9", "src_video"), ("10", "src_mask")):
        if job.get(key): wf[node]["inputs"]["video"] = os.path.basename(job[key])
        wf[node]["inputs"]["custom_width"], wf[node]["inputs"]["custom_height"] = w, h
        wf[node]["inputs"]["frame_load_cap"] = length
    if job.get("src_ref_images"): wf["12"]["inputs"]["image"] = os.path.basename(job["src_ref_images"])
    for nd in ("13", "14"): wf[nd]["inputs"]["width"], wf[nd]["inputs"]["height"] = w, h
    wf["14"]["inputs"]["length"] = length
    wf["15"]["inputs"]["steps"] = steps
    if job.get("prompt"): wf["7"]["inputs"]["text"] = job["prompt"]
    wf["18"]["inputs"]["filename_prefix"] = f"SLBENCH/{WORKER_ID}_{int(time.time())}"
    return wf, {"length": length, "steps": steps, "width": w, "height": h, "n_gpus": n}


def handler(event):
    job = (event or {}).get("input", {}) or {}
    if job.get("debug"):
        return _debug()
    if not job.get("prompt"):
        return {"error": "prompt is required"}

    cold = not _comfy_state["ready"]
    vram_entry = _vram_used()
    t_entry = time.time()
    if not _ensure_comfy():
        return {"error": "ComfyUI not available", "comfy": dict(_comfy_state),
                "comfy_log_tail": _tail(os.path.join(WDIR, "comfy.log")), "worker_id": WORKER_ID}

    n = int(job.get("n_gpus") or _n_gpus())
    jid = (event or {}).get("id") or f"{WORKER_ID}_{int(time.time())}"
    _beacon("req_entry", job_id=jid, cold=cold, n_gpus=n, vram_entry=vram_entry)
    wf, meta = _build_wf(job, n)
    t_submit = time.time()
    try:
        body = json.dumps({"prompt": wf, "client_id": str(jid)}).encode()
        req = urllib.request.Request(URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
        try:
            pid = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["prompt_id"]
        except urllib.error.HTTPError as e:
            return {"error": "workflow validation failed", "detail": e.read().decode()[:1500], "worker_id": WORKER_ID}
        err = None
        while time.time() - t_submit < 2400:
            time.sleep(3)
            try: h = _get(f"/history/{pid}")
            except Exception: continue
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") != "success": err = json.dumps(st.get("messages", []))[:1500]
                break
        t_done = time.time()
        vram_exit = _vram_used()
        mp4s = sorted(glob.glob(os.path.join(OUTPUT_DIR, wf["18"]["inputs"]["filename_prefix"] + "*.mp4")))
        out = {"worker_id": WORKER_ID, "epoch": MODULE_EPOCH, "job_id": jid, "cold": cold,
               "total_s": round(t_done - t_submit, 1), "vram_entry_mib": vram_entry, "vram_exit_mib": vram_exit, **meta}
        if err: out["error"] = "render failed"; out["detail"] = err
        elif mp4s:
            data = open(mp4s[-1], "rb").read(); out["bytes"] = len(data)
            out["video_base64"] = base64.b64encode(data).decode()
        else:
            out["error"] = "no output produced"
        _beacon("req_exit", job_id=jid, total_s=out["total_s"], ok=("error" not in out), vram_exit=vram_exit)
        return out
    except Exception:
        return {"error": "handler exception", "trace": traceback.format_exc(), "worker_id": WORKER_ID}


# The volume's ComfyUI model files are symlinks whose targets are /workspace/hf-cache/... (the POD mount point). On
# serverless the volume mounts at /runpod-volume AND /workspace already exists as a bare dir, so those symlinks dangle.
# Create /workspace/hf-cache -> /runpod-volume/hf-cache so every model symlink (gguf/lora/clip/vae, all under hf-cache/)
# resolves to the real blob. MUST run before ComfyUI loads anything.
try:
    os.makedirs("/workspace", exist_ok=True)
    _hf = "/workspace/hf-cache"
    if not os.path.lexists(_hf):
        os.symlink(os.path.join(VOL, "hf-cache"), _hf)
    log("workspace/hf-cache ->", os.path.realpath(_hf), "resolves:", os.path.exists(_hf))
except Exception as e:
    log("hf-cache link failed:", repr(e))

# --- module load: register the worker HEALTHY first, then warm ComfyUI in the background ---
log(f"boot worker={WORKER_ID} epoch={MODULE_EPOCH} tele={WDIR} on_volume={VOL_WRITABLE} "
    f"vol_exists={os.path.isdir(VOL)} comfy_dir_exists={os.path.isdir(COMFY_DIR)}")
_beacon("boot", on_volume=VOL_WRITABLE, vol_exists=os.path.isdir(VOL))
threading.Thread(target=_ensure_comfy, daemon=True).start()   # warm in background; NEVER blocks start()
runpod.serverless.start({"handler": handler})
