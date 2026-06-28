#!/usr/bin/env python3
"""RunPod Serverless handler — Raylight FSDP (2-GPU fp8 VACE + USP) render.

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
import os, sys, time, json, glob, base64, traceback, subprocess, threading, shutil, urllib.request, urllib.error


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
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{COMFY_DIR}/output")  # ComfyUI's DEFAULT — robust to --output-directory being ignored
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


def _hwtele(action, name=""):
    """FULL hardware telemetry via pod_telemetry.sh — the complete signal set (per-GPU sm%/mem-bw%/PCIe/NVLink/power/
    clocks/VRAM, CPU% + per-core, RAM, disk read/write, and NETWORK-volume RX rate, + per-process). The continuous
    net.csv (volume read) vs sys.csv/percpu.csv (CPU) split the cold load into VOLUME-READ vs Q4->bf16 DEQUANT — the
    thing the bare beacon couldn't show. Writes to WDIR/hw/. Crash-guarded; niced loggers never block the render."""
    try:
        script = os.path.join(HERE, "pod_telemetry.sh")
        if not os.path.exists(script):
            log("hwtele: pod_telemetry.sh missing at", script); return
        env = dict(os.environ); env["TELE_DIR"] = WDIR
        subprocess.Popen(["bash", script, action, "hw"] + ([name] if name else []),
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log("hwtele", action, "failed:", repr(e))


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
        "fp8":      os.path.join(base, "diffusion_models", "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors"),
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


def _comfy_dirs():
    """What ComfyUI will actually SCAN — confirms the model/input binds took (verify via cheap debug, pre-render)."""
    out = {}
    for sub in ("models", "models/vae", "models/text_encoders", "models/clip", "models/loras", "models/unet/gguf", "input"):
        p = os.path.join(COMFY_DIR, sub)
        try:
            top = os.path.join(COMFY_DIR, sub.split("/")[0])
            out[sub] = {"is_link": os.path.islink(top), "real": os.path.realpath(p) if os.path.lexists(p) else None,
                        "exists": os.path.isdir(p), "entries": sorted(os.listdir(p))[:10] if os.path.isdir(p) else None}
        except Exception as e:
            out[sub] = {"err": repr(e)}
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
    d["comfy_dirs"] = _comfy_dirs()
    if _comfy_state.get("phase") in ("error", "exception", "timeout"):
        d["comfy_log_tail"] = _tail(os.path.join(WDIR, "comfy.log"))
    return d


def _build_wf(job, n):
    wf = json.load(open(WF_PATH))
    length = int(job.get("frame_num", 81)); steps = int(job.get("sample_steps", 6))
    w, h = int(job.get("width", 720)), int(job.get("height", 1280))
    wf["1"]["inputs"]["GPU"] = n
    wf["1"]["inputs"]["ulysses_degree"] = n
    wf["1"]["inputs"]["FSDP"] = True
    wf["1"]["inputs"]["FSDP_CPU_OFFLOAD"] = False
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
    _hwtele("phase", f"req_entry:{jid}:cold={cold}:n={n}")
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
        _hwtele("phase", f"req_exit:{jid}:ok={'error' not in out}")
        _hwtele("save")
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

# ComfyUI must SCAN the volume's model + input trees. The --extra-model-paths-config / --input-directory launch args
# did NOT take effect (first render failed validation: empty model lists + "invalid input file"), so bind ComfyUI's
# OWN default dirs to the volume directly: replace the baked (empty placeholder) /opt/ComfyUI/{models,input} with
# symlinks to the volume. The volume's models/* entries are themselves symlinks into hf-cache, resolved by the
# /workspace/hf-cache link above. MUST run before the ComfyUI thread starts (i.e. before scanning).
def _bind(sub, target):
    try:
        link = os.path.join(COMFY_DIR, sub)
        if not os.path.isdir(target):
            log(f"bind {sub}: target MISSING {target}"); return
        if os.path.islink(link):
            log(f"bind {sub}: already a link -> {os.path.realpath(link)}"); return
        if os.path.exists(link):
            shutil.rmtree(link, ignore_errors=True)
        os.symlink(target, link)
        log(f"bind {sub} -> {os.path.realpath(link)} (entries={len(os.listdir(link))})")
    except Exception as e:
        log(f"bind {sub} failed:", repr(e))

# --- HOST-NVMe MODEL CACHE (the cold-load lever) ---------------------------------------------------------------
# The cold load is ~80% network-volume READ (~115s of ~140s on H100). RunPod's cached-model feature stages an HF
# repo onto host-local NVMe at /runpod/model-store/huggingface/<repo>/.../<file> (the native 75GB path's <1s mmap).
# So instead of one wholesale models->volume symlink, build a real models dir where EVERY model resolves from the
# volume by default (unchanged discovery + reads), but the 4 files the workflow loads PREFER the host-NVMe copy
# when RunPod has cached it. The net.csv->disk.csv read shift on the next cold render is the proof the cache works.
# SAFE FALLBACK: no cache registered => _hoststore returns None for all => behaviour identical to the old volume bind.
VOL_MODELS = os.path.join(VOL, "runpod-slim", "ComfyUI", "models")
MODELS_DIR = os.path.join(COMFY_DIR, "models")
# (ComfyUI-relative path under models/, HF repo that RunPod caches it from). Repos verified from the download manifest.
CACHED_MODELS = [
    ("diffusion_models/wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors", "Inner-Reflections/VACE_Skyreels_V3_R2V_Merge"),
    ("loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",                      "DeepBeepMeep/Wan2.1"),
    ("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",               "Comfy-Org/Wan_2.1_ComfyUI_repackaged"),
    ("vae/wan_2.1_vae.safetensors",                                        "Comfy-Org/Wan_2.1_ComfyUI_repackaged"),
]

def _hoststore(repo, basename):
    """Path to a RunPod-cached HF file on FAST storage, else None. RunPod surfaces the cache at two layouts seen in
    the wild: the native run logs used /runpod/model-store/huggingface/<repo>/<rev>/snapshots/<rev>/<file>; the
    current docs use the HF-hub form /runpod-volume/huggingface-cache/hub/models--<org>--<name>/snapshots/<rev>/<file>
    (same mount as the volume but fast-backed — 'loads significantly faster than a network volume'). Glob both by
    basename so we resolve it wherever RunPod put it; a miss falls back to the volume (no speedup, still correct)."""
    mangled = "models--" + repo.replace("/", "--")
    for root in (f"/runpod/model-store/huggingface/{repo}",
                 f"/runpod/model-store/huggingface/{mangled}",
                 f"/runpod-volume/huggingface-cache/hub/{mangled}"):
        hits = glob.glob(f"{root}/**/{basename}", recursive=True)
        if hits:
            return hits[0]
    return None

def _mirror_into(real_dir, vol_dir):
    """Make real_dir a real directory and symlink each entry of vol_dir into it, so all sibling models stay
    discoverable (the volume entries are themselves hf-cache symlinks = the network read = the fallback path)."""
    os.makedirs(real_dir, exist_ok=True)
    if os.path.isdir(vol_dir):
        for name in os.listdir(vol_dir):
            d = os.path.join(real_dir, name)
            if not os.path.lexists(d):
                os.symlink(os.path.join(vol_dir, name), d)

def _setup_models():
    """Per-file source selection for /opt/ComfyUI/models: volume by default; the 4 workflow models prefer host NVMe."""
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # default: wholesale-symlink every volume model subdir (full discovery, volume reads — old behaviour)
        if os.path.isdir(VOL_MODELS):
            for name in os.listdir(VOL_MODELS):
                d = os.path.join(MODELS_DIR, name)
                if not os.path.lexists(d):
                    os.symlink(os.path.join(VOL_MODELS, name), d)
        else:
            log("setup_models: VOL_MODELS MISSING", VOL_MODELS)
        # override the 4 workflow models: walk parents top-down, converting volume-symlinked dirs into real mirror
        # dirs so a single file can be replaced, then point the leaf at host NVMe when cached (else leave volume).
        for rel, repo in CACHED_MODELS:
            parts = rel.split("/")
            cur, volcur = MODELS_DIR, VOL_MODELS
            for comp in parts[:-1]:
                cur, volcur = os.path.join(cur, comp), os.path.join(volcur, comp)
                if os.path.islink(cur):
                    os.unlink(cur); _mirror_into(cur, volcur)
                elif not os.path.isdir(cur):
                    _mirror_into(cur, volcur)
            hs = _hoststore(repo, os.path.basename(rel))
            leaf = os.path.join(MODELS_DIR, *parts)
            if hs:
                if os.path.lexists(leaf):
                    os.unlink(leaf)
                os.symlink(hs, leaf)
            log(f"model {rel} <- {('HOST-NVMe ' + hs) if hs else 'volume (cache miss)'}")
        # CLIPLoader searches clip/, but the model lives in text_encoders/ on the volume.
        _cross_link_model("clip", "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                          os.path.join(MODELS_DIR, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
        # VAELoader expects the file directly in vae/, but the volume may have it inside vae/pixel_space/.
        _cross_link_model("vae", "wan_2.1_vae.safetensors",
                          os.path.join(VOL_MODELS, "vae", "pixel_space", "wan_2.1_vae.safetensors"))
    except Exception as e:
        log("setup_models failed:", repr(e))


def _cross_link_model(subdir, filename, source):
    """Ensure a model file is discoverable under MODELS_DIR/subdir/ by symlinking from source if needed."""
    target_dir = os.path.join(MODELS_DIR, subdir)
    target = os.path.join(target_dir, filename)
    if os.path.lexists(target) and os.path.exists(target):
        log(f"cross_link {subdir}/{filename}: already exists")
        return
    if not os.path.exists(source):
        for alt in glob.glob(os.path.join(MODELS_DIR, "**", filename), recursive=True):
            if os.path.exists(alt):
                source = alt; break
        else:
            log(f"cross_link {subdir}/{filename}: source NOT FOUND (tried {source})")
            return
    if os.path.islink(target_dir):
        vol_target = os.path.join(VOL_MODELS, subdir)
        os.unlink(target_dir)
        _mirror_into(target_dir, vol_target)
    os.makedirs(target_dir, exist_ok=True)
    if os.path.lexists(target):
        os.unlink(target)
    os.symlink(source, target)
    log(f"cross_link {subdir}/{filename} -> {source} (exists={os.path.exists(target)})")


_setup_models()
# Log model dir contents for debug visibility
for _sd in ("clip", "text_encoders", "vae", "diffusion_models", "loras"):
    _p = os.path.join(MODELS_DIR, _sd)
    try:
        _ents = sorted(os.listdir(_p))[:10] if os.path.isdir(_p) else f"MISSING(link={os.path.islink(_p)})"
        log(f"models/{_sd}: {_ents}")
    except Exception as _e:
        log(f"models/{_sd}: ERR {_e}")

_bind("input", INPUTS_DIR)
# Log input dir contents
try:
    _inp = os.path.join(COMFY_DIR, "input")
    _ents = sorted(os.listdir(_inp))[:15] if os.path.isdir(_inp) else f"NOT_DIR(exists={os.path.exists(_inp)},link={os.path.islink(_inp)})"
    log(f"input dir ({_inp} -> {os.path.realpath(_inp) if os.path.lexists(_inp) else 'N/A'}): {_ents}")
except Exception as _e:
    log(f"input dir: ERR {_e}")

# --- module load: register the worker HEALTHY first, then warm ComfyUI in the background ---
log(f"boot worker={WORKER_ID} epoch={MODULE_EPOCH} tele={WDIR} on_volume={VOL_WRITABLE} "
    f"vol_exists={os.path.isdir(VOL)} comfy_dir_exists={os.path.isdir(COMFY_DIR)}")
_beacon("boot", on_volume=VOL_WRITABLE, vol_exists=os.path.isdir(VOL))
_hwtele("start")   # full hw telemetry running BEFORE ComfyUI launches → captures the cold load (read vs dequant)
threading.Thread(target=_ensure_comfy, daemon=True).start()   # warm in background; NEVER blocks start()
runpod.serverless.start({"handler": handler})
