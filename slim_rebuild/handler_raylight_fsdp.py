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
import hashlib
import os, sys, time, json, glob, base64, traceback, subprocess, threading, shutil, urllib.request, urllib.error

# HOTPATCH RETIRED (pipeline contract §7, red-team H2, 2026-08-24): production
# code comes from the baked image ONLY. This file's base IS the volume's 7/16
# hotpatch (the last code that ran through that channel), pulled into git; the
# b2 comfy_dist files and raylight_nodes_r1.py are baked by the Dockerfile.


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


def _container_t0():
    """True container-start epoch from PID 1 (btime + starttime/HZ). _BOOT_T starts only after
    the interpreter + imports, so boot beacons under-count container age without this anchor."""
    try:
        with open("/proc/stat") as f:
            btime = next(int(l.split()[1]) for l in f if l.startswith("btime"))
        with open("/proc/1/stat") as f:
            st = f.read()
        ticks = float(st[st.rindex(")") + 1:].split()[19])
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


_CONTAINER_T0 = _container_t0()

# GLOBAL (2026-07-14, validated on stagingtest boot 4ykea5t0tti6ff): fire the Ray prestart
# immediately — the 3s settle delay ran serial after ComfyUI import and the warmup blocked 12.1s
# on the actor build; with delay=0 the build overlaps Comfy startup and the measured block was 0s
# (warmup ok, byte-path unchanged). Undo: delete this line or set RAYLIGHT_PRESTART_DELAY_S=3.
os.environ.setdefault("RAYLIGHT_PRESTART_DELAY_S", "0")


def _cpu_identity():
    """CPU model + host core count + THIS CONTAINER'S allowed cores (cpuset) for the boot
    beacon — per-claim CPU allocation becomes a checkable fact across boots."""
    model = None
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                model = l.split(":", 1)[1].strip(); break
    except Exception:
        pass
    try:
        allowed = len(os.sched_getaffinity(0))
    except Exception:
        allowed = None
    return {"cpu": model, "cores_host": os.cpu_count(), "cores_allowed": allowed}


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
            # preserve the PREVIOUS boot's log before we overwrite it — a hard-killed container's
            # final lines are the only crash-signature evidence (OOM cut-off vs driver trace).
            try:
                if os.path.exists(logp) and os.path.getsize(logp) > 0:
                    shutil.copyfile(logp, os.path.join(WDIR, f"comfy_prev_{int(time.time())}.log"))
            except Exception:
                pass
            try: out_f = open(logp, "wb", buffering=0)
            except Exception: out_f = None
            log(f"launching ComfyUI: COMFY_DIR={COMFY_DIR} args={cargs} gpus={env['CUDA_VISIBLE_DEVICES']}")
            # PIPE stdout through a per-line EPOCH stamper -> every ComfyUI/Ray/NCCL/render-node line gets
            # an absolute timestamp, so EVERY startup sub-phase + render node has a measurable duration
            # (the log was previously un-timestamped -> those durations were dark).
            _comfy = subprocess.Popen([sys.executable, os.path.join(HERE, "comfy_launch.py")],
                                      env=env, cwd=COMFY_DIR,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            def _ts_pipe(src, dst):
                try:
                    for line in iter(src.readline, b""):
                        try:
                            (dst.write(f"{time.time():.3f} ".encode() + line) if dst else None)
                            dst and dst.flush()
                        except Exception: pass
                except Exception: pass
            threading.Thread(target=_ts_pipe, args=(_comfy.stdout, out_f), daemon=True).start()
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
            time.sleep(0.25)   # was 2s: one poll cycle of dead time measured 0.77s on 7/13; ~0.1s expected now
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
        "env_runpod": {k: ("<redacted>" if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")) else v)
                       for k, v in os.environ.items() if k.startswith(("RUNPOD_", "MODEL_"))},
        "paths": {"VOL": VOL, "vol_exists": os.path.isdir(VOL),
                  "comfy_dir": COMFY_DIR, "comfy_dir_exists": os.path.isdir(COMFY_DIR),
                  "inputs_dir": INPUTS_DIR, "inputs_dir_exists": os.path.isdir(INPUTS_DIR),
                  "wf_path": WF_PATH, "wf_exists": os.path.exists(WF_PATH)},
    }
    try: d["vol_listing"] = sorted(os.listdir(VOL))[:25]
    except Exception as e: d["vol_listing"] = f"ERR {e}"
    d["models"] = _debug_models()
    d["comfy_dirs"] = _comfy_dirs()
    d["shard_prewarm"] = dict(_shard_prewarm_state)
    if _comfy_state.get("phase") in ("error", "exception", "timeout"):
        d["comfy_log_tail"] = _tail(os.path.join(WDIR, "comfy.log"))
    return d


def _io_probe():
    """Pin EXACTLY why the cached shard read is ~700 MB/s (3x slower than a single quiet 2013 read).
    Three controlled measurements on the SAME warm/cached file:
      (1) THREAD SCALING 1->16: rate RISING with threads = per-read FUSE round-trip LATENCY bound
          (parallelism hides it); rate FLAT = FUSE client BANDWIDTH capped.
      (2) CONCURRENCY: read shard alone vs while a 2nd thread hammers the OTHER shard = does the
          2-GPU/2-worker concurrency split the mfsmount throughput (the real loader case)?
      (3) MOUNT OPTIONS: /proc/self/mountinfo for the volume = the FS + cache mode.
    Pure I/O, no model load. Warms the file first so every timed read is from cache."""
    import glob as _g
    def _selfio():
        d = {}
        try:
            for l in open("/proc/self/io"):
                k, v = l.split(":"); d[k.strip()] = int(v)
        except Exception: pass
        return d
    def _pread(path, nthreads, stop=None):
        sz = os.path.getsize(path); buf = bytearray(sz); n = max(1, nthreads); chunk = (sz + n - 1)//n
        def rd(i):
            off = i*chunk; end = min(off+chunk, sz)
            if off >= end: return
            mv = memoryview(buf)[off:end]
            with open(path, "rb", buffering=0) as f:
                f.seek(off); got = 0
                while got < end-off:
                    if stop is not None and stop.is_set(): return
                    r = f.readinto(mv[got:])
                    if not r: break
                    got += r
        ts = [threading.Thread(target=rd, args=(i,)) for i in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        return sz
    def _timed(path, nt, stop=None):
        io0 = _selfio(); t0 = time.time(); sz = _pread(path, nt, stop=stop); dt = time.time()-t0; io1 = _selfio()
        return {"s": round(dt, 2), "MBps": round(sz/2**20/dt) if dt else 0,
                "read_bytes_gib": round((io1.get("read_bytes",0)-io0.get("read_bytes",0))/2**30, 2)}
    res = {"worker": WORKER_ID}
    try:
        res["mounts"] = [l for l in open("/proc/self/mountinfo").read().splitlines()
                         if "runpod-volume" in l or "fuse" in l.lower() or "moose" in l.lower()][:4]
    except Exception as e:
        res["mount_err"] = repr(e)[:120]
    shards = []
    for mp in sorted(_g.glob(os.path.join(VOL, "fsdp_shards_pr", ".prewarm_manifest_rank*"))):
        try:
            p = open(mp).read().strip()
            if os.path.isfile(p): shards.append(p)
        except Exception: pass
    if not shards: return {"err": "no manifest shard found", "mounts": res.get("mounts")}
    shard = shards[0]; other = shards[1] if len(shards) > 1 else shards[0]
    res["shard"] = shard; res["size_gib"] = round(os.path.getsize(shard)/2**30, 2)
    try:
        _pread(shard, 8)                                          # warm the file into cache first
        res["THREAD_SCALING_cached"] = {}
        for nt in (1, 2, 4, 8, 16, 32):
            res["THREAD_SCALING_cached"][f"{nt}t"] = _timed(shard, nt)   # rate vs thread count (cached)
        res["CONCURRENCY_solo_8t"] = _timed(shard, 8)            # shard alone, 8 threads
        st = threading.Event()
        def _bg():
            while not st.is_set(): _pread(other, 8, stop=st)     # 2nd reader on the OTHER shard = 2-GPU case
        th = threading.Thread(target=_bg); th.start(); time.sleep(0.3)
        res["CONCURRENCY_with_2nd_reader_8t"] = _timed(shard, 8)
        st.set(); th.join(timeout=15)
        smalls = []
        for rel in ("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "vae/wan_2.1_vae.safetensors",
                    "loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors"):
            p = os.path.realpath(os.path.join(VOL_MODELS, rel))
            if os.path.isfile(p): smalls.append(p)
        hammer_files = [other] + smalls
        st2 = threading.Event()
        def _hammer():
            while not st2.is_set():
                for hf in hammer_files:
                    if st2.is_set(): break
                    _pread(hf, 4, stop=st2)
        hts = [threading.Thread(target=_hammer) for _ in range(6)]
        for t in hts: t.start()
        time.sleep(0.5)
        res["SATURATED_read_8t"] = _timed(shard, 8)      # many concurrent mfs readers = full boot I/O
        st2.set()
        for t in hts: t.join(timeout=10)
    except Exception as e:
        res["probe_err"] = repr(e)[:200]
    return res


def _shard_ab(job):
    """COLD-read A/B on the SAME host: mmap loader (FSDP_SHARD_MMAP=1 path) vs legacy 8-stream
    reader, with page-cache EVICTION between reads (posix_fadvise DONTNEED) so every timed read
    re-fetches from the backing volume. Validity is measured, not assumed: /proc/self/io read_bytes
    must cover ~the file size (cold_valid) — a host/user-space cache that survives eviction shows
    up as backing_read_gib≈0 + cache-class rate and voids that arm. Waits for the boot warmup to
    finish first so reads run on a quiet worker. Rank shards discovered via the prewarm manifests
    (same as _io_probe). rep0 runs both rank shards, later reps shard[0] only (cost control)."""
    import glob as _g, io as _io
    import torch
    reps = int(job.get("reps", 2)); nthreads = int(job.get("threads", 8))
    res = {"worker": WORKER_ID, "arms": []}
    if job.get("wait_ready", 1):
        t0 = time.time()
        while time.time() - t0 < 420:            # warmup on a cold machine finishes ~+120s
            if max(_vram_used()) > 8000: break
            time.sleep(5)
        res["waited_ready_s"] = round(time.time() - t0, 1)
        time.sleep(3)
    shards = []
    for mp in sorted(_g.glob(os.path.join(VOL, "fsdp_shards_pr", ".prewarm_manifest_rank*"))):
        try:
            p = open(mp).read().strip()
            if os.path.isfile(p): shards.append(p)
        except Exception: pass
    if not shards: return {"err": "no manifest shards", "worker": WORKER_ID}
    def _evict(path):
        with open(path, "rb") as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    def _selfio2():
        d = {}
        try:
            for l in open("/proc/self/io"):
                k, v = l.split(":"); d[k.strip()] = int(v)
        except Exception: pass
        return d
    def _touch_pages(obj):
        """Force page-in of every tensor, dtype-proof: view the UNTYPED storage as uint8 and read
        one byte per 4KiB page (what .to(device) would fault in, without the copy)."""
        seen, total = set(), 0
        stack = [obj]
        while stack:
            x = stack.pop()
            if torch.is_tensor(x):
                s = x.untyped_storage()
                if s.data_ptr() in seen: continue
                seen.add(s.data_ptr())
                u8 = torch.empty(0, dtype=torch.uint8); u8.set_(s)
                if u8.numel(): total += int(u8[::4096].sum().item())
            elif isinstance(x, dict): stack.extend(x.values())
            elif isinstance(x, (list, tuple)): stack.extend(x)
            elif hasattr(x, "__dict__"): stack.extend(vars(x).values())
        return total
    def _arm_mmap(path):
        t0 = time.time()
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        t_open = time.time() - t0
        _touch_pages(obj)                        # the page-in IS the read on this arm
        return {"t_open": round(t_open, 2)}
    def _arm_8stream(path):
        sz = os.path.getsize(path); buf = bytearray(sz); chunk = (sz + nthreads - 1)//nthreads
        t0 = time.time()
        def rd(i):
            off = i*chunk; end = min(off+chunk, sz)
            if off >= end: return
            mv = memoryview(buf)[off:end]
            with open(path, "rb", buffering=0) as f:
                f.seek(off); got = 0
                while got < end-off:
                    r = f.readinto(mv[got:])
                    if not r: break
                    got += r
        ts = [threading.Thread(target=rd, args=(i,)) for i in range(nthreads)]
        [t.start() for t in ts]; [t.join() for t in ts]
        t_read = time.time() - t0
        torch.load(_io.BytesIO(buf), map_location="cpu", weights_only=False)   # legacy eager unpickle
        del buf
        return {"t_read": round(t_read, 2)}
    try:
        for rep in range(reps):
            for arm_name, fn in (("mmap", _arm_mmap), ("8stream", _arm_8stream)):
                for path in (shards if rep == 0 else shards[:1]):
                    sz_gib = os.path.getsize(path)/2**30
                    _evict(path)
                    io0 = _selfio2(); t0 = time.time()
                    r = fn(path)
                    dt = time.time() - t0; io1 = _selfio2()
                    rb = (io1.get("read_bytes", 0) - io0.get("read_bytes", 0))/2**30
                    res["arms"].append({"rep": rep, "arm": arm_name, "shard": os.path.basename(path)[-24:],
                                        "size_gib": round(sz_gib, 2), "t_total_s": round(dt, 2),
                                        "rate_MBps": round(sz_gib*1024/dt) if dt else 0,
                                        "backing_read_gib": round(rb, 2), "cold_valid": rb > 0.8*sz_gib, **r})
        for arm_name, fn in (("mmap_WARM", _arm_mmap), ("8stream_WARM", _arm_8stream)):   # no-evict baselines
            t0 = time.time(); fn(shards[0]); dt = time.time() - t0
            res["arms"].append({"arm": arm_name, "shard": os.path.basename(shards[0])[-24:],
                                "t_total_s": round(dt, 2),
                                "rate_MBps": round(os.path.getsize(shards[0])/2**20/dt) if dt else 0})
    except Exception as e:
        res["ab_err"] = repr(e)[:300]
        res["trace"] = traceback.format_exc()[-800:]
    return res


_MODEL_SOURCES = {}   # rel -> "HOST-NVMe <path>" | "volume"; filled by _setup_models before the prewarm thread starts


def _prewarm_fileset():
    """The exact file set the boot prewarm warms: manifest shards (fallback: newest rank_*.pt
    per basename) plus the small render models. Shared by the boot prewarm and _prewarm_probe.
    NVMe-AWARE (2026-07-14): a file whose effective load source is host NVMe is SKIPPED — the
    loader never reads its volume copy, so warming it only burns volume bandwidth (7/14: this
    wasted 23 GB read ran beside a degraded volume mount). Files still served FROM the volume
    (no staging on this host, cache miss) are warmed exactly as before — the prewarm is the
    fallback path, not deleted. PREWARM_SKIP_NVME=0 restores warm-everything."""
    skip_nvme = os.environ.get("PREWARM_SKIP_NVME", "1") == "1"
    skipped = []
    sdir = os.environ.get("FSDP_SHARD_DIR", os.path.join(VOL, "fsdp_shards_pr"))
    manifest = []
    for mp in glob.glob(os.path.join(sdir, ".prewarm_manifest_rank*")):
        try:
            fp = open(mp).read().strip()
            if fp and os.path.isfile(fp):
                manifest.append(fp)
        except Exception:
            pass
    if manifest:
        files, src = sorted(set(manifest)), "manifest"
    else:
        by_name = {}
        for p in glob.glob(os.path.join(sdir, "*", "rank_*.pt")):
            k = os.path.basename(p)
            if k not in by_name or os.path.getmtime(p) > os.path.getmtime(by_name[k]):
                by_name[k] = p
        files, src = sorted(by_name.values()), "glob_fallback"
    if skip_nvme and os.environ.get("HOSTSTORE_SHARDS_DIR") and files:
        skipped.append(f"shards x{len(files)} (loader reads HOSTSTORE_SHARDS_DIR)")
        files, src = [], src + "+shards_nvme_skipped"
    for rel in ("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "vae/wan_2.1_vae.safetensors",
                "loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors"):
        if skip_nvme and _MODEL_SOURCES.get(rel, "").startswith("HOST-NVMe"):
            skipped.append(rel)
            continue
        p = os.path.realpath(os.path.join(VOL_MODELS, rel))
        if os.path.isfile(p):
            files.append(p)
    if skipped:
        log(f"prewarm: NVMe-served, not warmed from volume: {'; '.join(skipped)}; "
            f"volume-warm list = {len(files)} file(s)")
    _beacon("prewarm_plan", src=src, files=len(files), skipped=skipped)
    return files, src


def _prewarm_jobs(files, chunk=64 * 1024 * 1024):
    """Build the prewarm chunk queue in NEED ORDER + shard-interleaved (7/11 boot forensics,
    worker x8imf1x3yqsbau): the warmup loads TE/VAE/lora at t≈7-45s and the FSDP shards at
    t≈48s, but the old queue read shards FIRST — under memory pressure the oldest pages get
    evicted first, so rank_0 (read first, needed last) was 17.7% resident at load time and
    cost a 40s 8stream re-read of a file the prewarm had ALREADY read; rank_1 (read last)
    was 100% resident -> 1.9s mmap. So: small render models first (needed first, freshest
    least matters), then the rank shards with chunks INTERLEAVED round-robin — both GPU
    ranks mmap their shards SIMULTANEOUSLY, and two-file spread measured ~2x aggregate
    throughput vs single-file draining (dual_read: 1049 vs ~360 MB/s single)."""
    shards = [p for p in files if os.path.basename(p).startswith("rank_")]
    smalls = [p for p in files if p not in shards]
    jobs = []
    total = 0
    for p in smalls:
        sz = os.path.getsize(p); total += sz
        for off in range(0, sz, chunk):
            jobs.append((p, off, min(chunk, sz - off)))
    per_shard = []
    for p in shards:
        sz = os.path.getsize(p); total += sz
        per_shard.append([(p, off, min(chunk, sz - off)) for off in range(0, sz, chunk)])
    idx = 0
    remaining = True
    while remaining:
        remaining = False
        for lst in per_shard:
            if idx < len(lst):
                jobs.append(lst[idx]); remaining = True
        idx += 1
    return jobs, total


def _adaptive_chunk_read(jobs, start_workers=8, min_workers=4, max_workers=16,
                         window_s=1.5, gain=1.10, tune_floor_bytes=2 * 2**30, stall_s=120):
    """Drain (path, offset, length) chunk jobs with a self-tuning thread pool.
    The right thread count is PER-HOST (7/11 same-file eviction A/B: 8t ~600 vs 16t ~830 MB/s
    on one host; an earlier host had 8t 978 > 16t 932; cached ceiling ~1.8 GB/s flat 8..32).
    Ladder: measure a window at the current count (first window after any change is discarded
    as ramp), step up 2x; a step is kept only after TWO consecutive windows beat the base by
    >10% (cold-read rates drift upward as readahead warms — one window is not a signal), else
    revert one step and lock. Default max 16: cold gains past 16 are unmeasured and >6 readers
    measured a collapse regime — probes may pass higher explicitly. Tuning only runs while
    >tune_floor_bytes remain, so short tails never mistune. A frozen byte counter for stall_s
    aborts (wedged mount) with stalled=True rather than spinning forever. Returns totals +
    the tuning trace; per-chunk read errors are counted, never raised — the CALLER must check
    bytes vs planned and stalled, not just the absence of an exception."""
    import queue as _q
    q = _q.Queue()
    total = 0
    for j in jobs:
        q.put(j); total += j[2]
    done = [0]; errs = [0]; lk = threading.Lock()
    target = [max(min_workers, min(start_workers, max_workers))]
    threads = []

    def _worker(idx):
        while idx < target[0]:
            try:
                p, off, ln = q.get_nowait()
            except _q.Empty:
                return
            try:
                with open(p, "rb", buffering=0) as f:
                    f.seek(off)
                    left = ln
                    while left > 0:
                        b = f.read(min(8 * 1024 * 1024, left))
                        if not b:
                            break
                        left -= len(b)
                        with lk:
                            done[0] += len(b)
            except Exception:
                with lk:
                    errs[0] += 1

    def _ensure(n):
        while len(threads) < n:
            t = threading.Thread(target=_worker, args=(len(threads),), daemon=True)
            t.start(); threads.append(t)

    t0 = time.time()
    _ensure(target[0])
    trace = []
    base = None                 # accepted rate for the current locked-in worker count
    prev_w = target[0]
    locked = False
    discard = True              # first window after any change measures ramp -> discard
    hits = 0                    # consecutive candidate windows beating base (need 2)
    stalled_for = 0.0
    last_b, last_t = 0, t0
    while any(t.is_alive() for t in threads):
        time.sleep(window_s)
        now = time.time(); b = done[0]
        rate = (b - last_b) / max(now - last_t, 1e-6) / 2**20
        frozen = b == last_b
        last_b, last_t = b, now
        if len(trace) < 24:
            trace.append({"t": round(now - t0, 1), "w": target[0], "MBps": round(rate)})
        if frozen:
            stalled_for += window_s
            if stalled_for >= stall_s:
                break                         # wedged mount: bail, don't spin forever
            continue
        stalled_for = 0.0
        if locked or (total - b) < tune_floor_bytes:
            continue
        if discard:
            discard = False
            continue
        if base is None:
            base = rate                       # baseline at start_workers
        elif rate > base * gain:
            hits += 1
            if hits < 2:
                continue                      # drift guard: one hot window is not a signal
            base = rate; hits = 0             # step confirmed twice -> new baseline
        else:
            if target[0] != prev_w:
                target[0] = prev_w            # candidate failed -> revert one step
            locked = True
            hits = 0
            continue
        if target[0] < max_workers:
            prev_w = target[0]
            target[0] = min(target[0] * 2, max_workers)
            _ensure(target[0])
            discard = True
        else:
            locked = True
    dur = max(time.time() - t0, 1e-6)
    return {"bytes": done[0], "dur_s": round(dur, 1),
            "MBps_avg": round(done[0] / 2**20 / dur), "chosen_w": target[0],
            "tail_unmeasured": bool(discard and not locked),
            "stalled": stalled_for >= stall_s,
            "trace": trace, "errors": errs[0], "leftover": q.qsize()}


def _probe_quiesce(res, opts):
    """Shared guard for the I/O probe hooks (same reason as _shard_ab's wait_ready: reads
    measured DURING boot run concurrent with the boot prewarm + warmup load and are garbage).
    Waits for warmup VRAM unless opts['wait_ready']=0. NOTE these probes then EVICT the shard
    set from page cache — on a warm prod worker the next reload reads cold; that is the point
    of the probe, but it is a real cost: probe deliberately, not casually."""
    if int(opts.get("wait_ready", 1)):
        t0 = time.time()
        while time.time() - t0 < 420:
            if max(_vram_used()) > 8000:
                break
            time.sleep(5)
        res["waited_ready_s"] = round(time.time() - t0, 1)
        time.sleep(3)


def _prewarm_probe(opts):
    """Job-driven validation of the adaptive prewarm on THIS worker: quiesce, evict the
    prewarm file set from page cache (default), drain it via _adaptive_chunk_read, return
    the tuning trace. opts: {'evict':1, 'start_workers':8, 'max_workers':16, 'wait_ready':1}."""
    res = {"worker": WORKER_ID}
    try:
        _probe_quiesce(res, opts)
        files, src = _prewarm_fileset()
        res["src"] = src
        res["files"] = [os.path.basename(p)[-28:] for p in files]
        if not files:
            return {**res, "err": "no files"}
        if int(opts.get("evict", 1)):
            for p in files:
                with open(p, "rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        jobs, _tot = _prewarm_jobs(files)
        start_w = max(1, min(int(opts.get("start_workers",
                                          os.environ.get("SHARD_CACHE_PREWARM_THREADS", "8"))), 64))
        max_w = max(start_w, min(int(opts.get("max_workers", 16)), 64))
        res.update(_adaptive_chunk_read(jobs, start_workers=start_w, max_workers=max_w))
    except Exception as e:
        res["err"] = repr(e)[:300]
        res["tb"] = traceback.format_exc()[-600:]
    return res


def _dual_read(opts):
    """Per-file vs per-mount ceiling discriminator, self-contained (both arms in ONE run,
    same host, same moment, same per-file thread count so reader-count collapse can't
    confound the comparison):
      arm SOLO: evict shard A, read it alone with N range-threads          -> rate S
      arm DUAL: evict A and B, read both concurrently, N threads EACH      -> rates Da, Db
    Per-mount shared cap -> Da ~ S/2. Per-file cap -> Da ~ S. Default N=8 (the measured
    baseline count). opts: {'threads':8, 'wait_ready':1}."""
    nthreads = max(1, min(int(opts.get("threads", 8)), 32))
    res = {"worker": WORKER_ID, "threads_each": nthreads}
    try:
        _probe_quiesce(res, opts)
        sdir = os.environ.get("FSDP_SHARD_DIR", os.path.join(VOL, "fsdp_shards_pr"))
        shards = []
        for mp in sorted(glob.glob(os.path.join(sdir, ".prewarm_manifest_rank*"))):
            try:
                p = open(mp).read().strip()
                if os.path.isfile(p):
                    shards.append(p)
            except Exception:
                pass
        if len(shards) < 2:
            return {**res, "err": f"need 2 manifest shards, have {len(shards)}"}
        a, b = shards[0], shards[1]

        def _evict(paths):
            for p in paths:
                with open(p, "rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

        def _read_range_threads(path, out, key):
            try:
                sz = os.path.getsize(path)
                n = nthreads; chunk = (sz + n - 1) // n
                fail = [0]
                def rd(i):
                    off = i * chunk; end = min(off + chunk, sz)
                    if off >= end:
                        return
                    try:
                        with open(path, "rb", buffering=0) as f:
                            f.seek(off); left = end - off
                            while left > 0:
                                r = f.read(min(8 * 1024 * 1024, left))
                                if not r:
                                    break
                                left -= len(r)
                    except Exception:
                        fail[0] += 1
                t0 = time.time()
                ts = [threading.Thread(target=rd, args=(i,)) for i in range(n)]
                [t.start() for t in ts]; [t.join() for t in ts]
                dt = max(time.time() - t0, 1e-6)
                out[key] = {"s": round(dt, 2), "MBps": round(sz / 2**20 / dt),
                            "size_gib": round(sz / 2**30, 2), "thread_errs": fail[0]}
            except Exception as e:
                out[key] = {"err": repr(e)[:200]}

        out = {}
        _evict([a])                                            # arm SOLO
        _read_range_threads(a, out, "solo_a")
        _evict([a, b])                                         # arm DUAL
        t0 = time.time()
        ta = threading.Thread(target=_read_range_threads, args=(a, out, "dual_a"))
        tb = threading.Thread(target=_read_range_threads, args=(b, out, "dual_b"))
        ta.start(); tb.start(); ta.join(); tb.join()
        wall = max(time.time() - t0, 1e-6)
        tot = os.path.getsize(a) + os.path.getsize(b)
        res.update(out)
        res["dual_wall_s"] = round(wall, 2)
        res["dual_aggregate_MBps"] = round(tot / 2**20 / wall)
        try:
            res["dual_over_solo"] = round(out["dual_a"]["MBps"] / max(out["solo_a"]["MBps"], 1), 2)
        except Exception:
            pass
    except Exception as e:
        res["err"] = repr(e)[:300]
        res["tb"] = traceback.format_exc()[-600:]
    return res


def _build_wf(job, n):
    wf = json.load(open(WF_PATH))
    # Defaults = Anna's validated pipeline settings (PIPELINE_GUIDE_FOR_PARTNER.md, 2026-07-03):
    # 576p-class render res (aspect MUST match source orientation), res_2s/beta57/8, cfg 1.0 (FusionX).
    length = int(job.get("frame_num", 81)); steps = int(job.get("sample_steps", 8))
    w, h = int(job.get("width", 576)), int(job.get("height", 1024))
    wf["1"]["inputs"]["GPU"] = n
    wf["1"]["inputs"]["ulysses_degree"] = n
    # FSDP switchable per job. Anna's env errors FSDP+fp8 on A100 ("FP8 reduction ... sm90") but our
    # stack verified it on 2xA100 SXM 6/27 — unresolved env difference, see PROJECT_HISTORY Phase 9.
    wf["1"]["inputs"]["FSDP"] = bool(job.get("fsdp", True))
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
    wf["15"]["inputs"]["cfg"] = float(job.get("cfg", 1.0))
    if job.get("sampler_name"): wf["15"]["inputs"]["sampler_name"] = job["sampler_name"]
    if job.get("scheduler"): wf["15"]["inputs"]["scheduler"] = job["scheduler"]
    # strength (VACE control adherence, node 14) and shift (sigma schedule, node 4): sent by the
    # backend per look mode (Anna's recipes: photoreal 1.0/6, 2d 0.85/8). IGNORING these was the
    # 8/23 red-team M7 finding — her 2d mode silently rendered with photoreal settings. Applied
    # and echoed (params_effective) so the backend can assert what actually ran (contract §1).
    if job.get("strength") is not None: wf["14"]["inputs"]["strength"] = float(job["strength"])
    if job.get("shift") is not None: wf["4"]["inputs"]["shift"] = float(job["shift"])
    if job.get("prompt"): wf["7"]["inputs"]["text"] = job["prompt"]
    # 2D/cartoon targets need the default negative REPLACED (it bans "cartoon, anime, ..." which
    # fights flat styles; swap in "photorealistic, 3d render, ..." per the pipeline guide).
    if job.get("negative_prompt") is not None: wf["8"]["inputs"]["text"] = job["negative_prompt"]
    wf["18"]["inputs"]["filename_prefix"] = f"SLBENCH/{WORKER_ID}_{int(time.time())}"
    # Echo of every output-affecting setting AS APPLIED, read back from the workflow itself
    # (ruled 8/24: all settings backend-controlled; worker built-ins never load-bearing).
    effective = {"sample_steps": steps, "cfg": wf["15"]["inputs"]["cfg"],
                 "sampler_name": wf["15"]["inputs"]["sampler_name"], "scheduler": wf["15"]["inputs"]["scheduler"],
                 "strength": wf["14"]["inputs"]["strength"], "shift": wf["4"]["inputs"]["shift"],
                 "frame_num": length, "width": w, "height": h,
                 "prompt": wf["7"]["inputs"]["text"], "negative_prompt": wf["8"]["inputs"]["text"]}
    return wf, {"length": length, "steps": steps, "width": w, "height": h, "n_gpus": n,
                "cfg": wf["15"]["inputs"]["cfg"], "fsdp": wf["1"]["inputs"]["FSDP"],
                "params_effective": effective}


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fetch_input(url, path, expected):
    """Presigned-GET download with Range resume (contract §4). Returns bytes
    fetched; raises HTTPError on 401/403 (expired URL — retrying cannot help)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    got = 0
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url)
            if got:
                req.add_header("Range", f"bytes={got}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if (got and getattr(r, "status", 200) == 206) else "wb"
                if mode == "wb":
                    got = 0
                with open(tmp, mode) as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise
            log(f"input fetch attempt {attempt}: HTTP {e.code}")
        except Exception as e:
            log(f"input fetch attempt {attempt}: {type(e).__name__}: {e}")
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expected is None or got == expected:
            break
        time.sleep(2 * attempt)
    if expected is None or got == expected:
        os.replace(tmp, path)
    return got


def _put_result(url, data, attempts=3):
    """Upload the mp4 to the presigned PUT URL (contract §4: 3 attempts, 2 s/4 s
    backoff). Returns the ETag; raises after the last failure."""
    last = None
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, data=data, method="PUT",
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))})
            with urllib.request.urlopen(req, timeout=600) as r:
                return (r.headers.get("ETag") or "").strip('"')
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise
            last = f"HTTP {e.code}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        log(f"result PUT attempt {i} failed: {last}")
        if i < attempts:
            time.sleep(2 * i)
    raise RuntimeError(f"PUT failed after {attempts} attempts: {last}")


def handler(event):
    job = (event or {}).get("input", {}) or {}
    if job.get("env_probe"):
        # cheap pre-flight: run proc_sampler in preflight mode -> returns EXACTLY what is readable on
        # this pod (cgroup version, ptrace/kmsg/fuse/perf caps, real sched keys, netns visibility) so
        # we verify the one-shot capture will have data BEFORE spending it. No model load needed.
        try:
            d = os.path.join(WDIR, "pf")
            subprocess.run([sys.executable, os.path.join(HERE, "proc_sampler.py"), d, "preflight"], timeout=90)
            return json.loads(open(os.path.join(d, "boot.json")).read()).get("PREFLIGHT", {"err": "no PREFLIGHT"})
        except Exception as e:
            return {"env_probe_err": repr(e)[:300]}
    if job.get("io_probe"):
        return _io_probe()
    if job.get("shard_ab") is not None:
        return _shard_ab(job["shard_ab"] if isinstance(job["shard_ab"], dict) else {})
    if job.get("prewarm_probe") is not None:
        return _prewarm_probe(job["prewarm_probe"] if isinstance(job["prewarm_probe"], dict) else {})
    if job.get("dual_read") is not None:
        return _dual_read(job["dual_read"] if isinstance(job["dual_read"], dict) else {})
    if job.get("debug"):
        # hold_sec: keep this job EXECUTING for N seconds (capped 120) — instrument for testing
        # whether billed execution time resets RunPod's idle-container kill timer.
        try:
            hold = min(float(job.get("hold_sec") or 0), 120.0)
        except Exception:
            hold = 0.0
        if hold > 0:
            time.sleep(hold)
        d = _debug()
        if job.get("freeze"):
            # runtime-matrix capture for the slim-image rebuild: exact package set + CUDA
            # userspace libs of the LIVE container (the checked-in Dockerfile is not the
            # deployed artifact's source of truth).
            d["pip_freeze"] = _run([sys.executable, "-m", "pip", "freeze", "--all"], timeout=120)
            d["ldconfig_cuda"] = "\n".join(l for l in _run(["ldconfig", "-p"], timeout=60).splitlines()
                                           if any(s in l.lower() for s in
                                                  ("cudnn", "cublas", "nccl", "cudart", "cufft",
                                                   "curand", "cusparse", "cusolver", "nvjitlink", "nvrtc")))
            d["driver"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=30)
        return d
    # Contract v1 (onset docs/design/pipeline-contract.md §3): inputs arrive as
    # presigned URLs with bytes+sha256 (verified BEFORE any GPU work), the
    # result goes back by presigned PUT — no inline base64, no volume paths,
    # no 6 MB cap. Legacy jobs (no contract_version) keep the old transport.
    contract = job.get("contract_version") is not None
    t_fetch0 = time.time()
    if contract:
        if job.get("contract_version") != 1:
            return {"error": "CONTRACT_MISMATCH", "detail": f"worker speaks contract 1, got {job.get('contract_version')!r}"}
        if not isinstance(job.get("result"), dict) or not job["result"].get("put_url"):
            return {"error": "CONTRACT_MISMATCH", "detail": "result.put_url is required"}
        for role, spec in (job.get("inputs") or {}).items():
            name = spec.get("name") or ""
            if not name or "/" in name or "\\" in name or ".." in name:
                return {"error": "FETCH_VERIFY_FAILED", "detail": f"{role}: name must be a bare filename, got {name!r}"}
            path = os.path.join(INPUTS_DIR, name)
            try:
                got = _fetch_input(spec["url"], path, spec.get("bytes"))
            except urllib.error.HTTPError as e:
                return {"error": "URL_EXPIRED" if e.code in (401, 403) else "FETCH_VERIFY_FAILED",
                        "detail": f"{role}: HTTP {e.code}"}
            if spec.get("bytes") is not None and got != spec["bytes"]:
                return {"error": "FETCH_VERIFY_FAILED", "detail": f"{role}: {got}/{spec['bytes']} bytes"}
            if spec.get("sha256") and _sha256_file(path) != spec["sha256"]:
                return {"error": "FETCH_VERIFY_FAILED", "detail": f"{role}: sha256 mismatch"}
            job[role] = name  # downstream workflow build reads the filename

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
        # Contract §5: the leg budget arrives with the job (backend derives it
        # from the 30-min ceiling); legacy jobs keep the old 40-min cap.
        budget_s = float(job.get("budget_s") or 2400)
        while time.time() - t_submit < budget_s:
            time.sleep(3)
            try: h = _get(f"/history/{pid}")
            except Exception: continue
            if pid in h:
                st = h[pid].get("status", {})
                # Keep the TAIL: ComfyUI's messages end with the exception; the old head-cut at
                # 1,500 chars dropped the exception class on every 9/03 single-H100 failure.
                if st.get("status_str") != "success": err = json.dumps(st.get("messages", []))[-6000:]
                break
        t_done = time.time()
        vram_exit = _vram_used()
        mp4s = sorted(glob.glob(os.path.join(OUTPUT_DIR, wf["18"]["inputs"]["filename_prefix"] + "*.mp4")))
        out = {"worker_id": WORKER_ID, "epoch": MODULE_EPOCH, "job_id": jid, "cold": cold,
               "total_s": round(t_done - t_submit, 1), "vram_entry_mib": vram_entry, "vram_exit_mib": vram_exit, **meta}
        if err: out["error"] = "render failed"; out["detail"] = err
        elif mp4s and contract:
            # Contract §3/§4: the result goes straight to object storage; size
            # is unbounded (the >6 MB inline failure class is gone by design).
            data = open(mp4s[-1], "rb").read(); out["bytes"] = len(data)
            t_put0 = time.time()
            try:
                etag = _put_result(job["result"]["put_url"], data)
            except urllib.error.HTTPError as e:
                out["error"] = "URL_EXPIRED" if e.code in (401, 403) else "PUT_FAILED"; out["detail"] = f"HTTP {e.code}"
            except Exception as e:
                out["error"] = "PUT_FAILED"; out["detail"] = str(e)[:300]
            else:
                out["contract_version"] = 1
                out["result"] = {"key": job["result"].get("key"), "bytes": len(data),
                                 "sha256": hashlib.sha256(data).hexdigest(), "etag": etag}
                out["timings"] = {"fetch_s": round(t_submit - t_fetch0, 1),
                                  "render_s": round(t_done - t_submit, 1),
                                  "put_s": round(time.time() - t_put0, 1)}
        elif mp4s:
            data = open(mp4s[-1], "rb").read(); out["bytes"] = len(data)
            # RunPod response payloads have hard size limits; base64 adds +33%. Real-length clips
            # (112-360 f => ~8-30 MB) must go via the volume instead of inline. Threshold in MB,
            # override with VIDEO_INLINE_MAX_MB.
            inline_max = float(os.environ.get("VIDEO_INLINE_MAX_MB", "6")) * 1024 * 1024
            if len(data) <= inline_max:
                out["video_base64"] = base64.b64encode(data).decode()
            else:
                rel = f"outputs/{WORKER_ID}_{jid}.mp4"
                vol_path = os.path.join(VOL, rel)
                os.makedirs(os.path.dirname(vol_path), exist_ok=True)
                with open(vol_path, "wb") as f:
                    f.write(data)
                out["video_volume_path"] = rel   # fetch via S3 GET on the volume bucket
                out["video_inline"] = False
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
# All production models consolidated in one HF repo for RunPod model caching.
CACHE_REPO = "Chicolll/bg-replace-pipeline"
CACHED_MODELS = [
    ("diffusion_models/wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors", CACHE_REPO),
    ("loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",                      CACHE_REPO),
    ("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",               CACHE_REPO),
    ("vae/wan_2.1_vae.safetensors",                                        CACHE_REPO),
    ("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",                CACHE_REPO),
    ("unet/qwen-image-edit-2511-Q5_0.gguf",                                CACHE_REPO),
    ("loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",  CACHE_REPO),
    ("vae/qwen_image_vae.safetensors",                                     CACHE_REPO),
    ("sam3/sam3.pt",                                                        CACHE_REPO),
]

def _hoststore(repo, rel_path):
    """Path to a RunPod-cached HF file on FAST storage, else None."""
    basename = os.path.basename(rel_path)
    # Direct hit at the documented model-cache path (set via endpoint modelName field).
    direct = f"/runpod/cache/{repo}/main/{rel_path}"
    if os.path.isfile(direct):
        return direct
    # Fallback: glob older RunPod cache layouts by basename.
    mangled = "models--" + repo.replace("/", "--")
    for root in (f"/runpod/model-store/huggingface/{repo}",
                 f"/runpod/model-store/huggingface/{mangled}",
                 f"/runpod-volume/huggingface-cache/hub/{mangled}",
                 f"/runpod/cache/{repo}"):
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
    _sm_t0 = time.time()
    # hoststore probe (2026-07-09): every cold boot logs "cache miss" — but WHY is invisible: does
    # the host store not exist on our hosts, exist-but-empty, or exist with an unexpected layout?
    # One boot with this logging answers it. Free passenger on every boot.
    try:
        _hs_seen = {}
        # roots per official docs (2026-07-12, docs.runpod.io/serverless/endpoints/model-caching):
        # cached models surface at /runpod-volume/huggingface-cache/hub/ — probe BOTH conventions.
        for _hsroot in ("/runpod", "/runpod/model-store", "/runpod/model-store/huggingface", "/runpod/cache",
                        "/runpod-volume/huggingface-cache", "/runpod-volume/huggingface-cache/hub"):
            if os.path.isdir(_hsroot):
                _hs_seen[_hsroot] = sorted(os.listdir(_hsroot))[:12]
                log(f"hoststore: {_hsroot} -> {_hs_seen[_hsroot]}")
            else:
                _hs_seen[_hsroot] = None
                log(f"hoststore: {_hsroot} MISSING")
        # stdout is console-only; mirror to the volume so the verdict survives without the console
        _beacon("hoststore_probe", roots=_hs_seen)
        # staged-store SPEED probe (test endpoint only — never in prod's boot path): evict, then
        # 8-thread read of the first 2 GiB of the staged UNet. Runs before the prewarm thread
        # spawns, so the measurement is clean. NVMe-class vs network-class is the go/no-go for
        # wiring the load path onto the host store.
        if os.environ.get("RUNPOD_ENDPOINT_ID") == "69qtffutk83l3o":
            _p = ("/runpod/model-store/huggingface/Chicolll/bg-replace-pipeline/"
                  "876036aae292599e10e514bfb1f6c99088a86497/snapshots/"
                  "876036aae292599e10e514bfb1f6c99088a86497/diffusion_models/"
                  "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors")
            if os.path.isfile(_p):
                import concurrent.futures as _cf
                _fd = os.open(_p, os.O_RDONLY)
                try: os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally: os.close(_fd)
                _N, _CH = 8, 2 * 1024**3
                def _rd(i):
                    with open(_p, "rb", buffering=0) as f:
                        f.seek(i * (_CH // _N)); n = 0
                        while n < _CH // _N:
                            b = f.read(min(16 * 1024**2, _CH // _N - n))
                            if not b: break
                            n += len(b)
                    return n
                _t0 = time.time()
                with _cf.ThreadPoolExecutor(_N) as _ex:
                    _tot = sum(_ex.map(_rd, range(_N)))
                _dt = max(time.time() - _t0, 1e-6)
                _beacon("hoststore_read_rate", gib=round(_tot / 1024**3, 2), sec=round(_dt, 2),
                        gbps=round(_tot / 1024**3 / _dt, 2))
    except Exception as _e:
        log("hoststore probe failed:", repr(_e))
    _src = {}
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # default: wholesale-symlink every volume model subdir (full discovery, volume reads — old behaviour)
        if os.path.isdir(VOL_MODELS):
            for name in os.listdir(VOL_MODELS):
                d = os.path.join(MODELS_DIR, name)
                # fresh-image stock dirs (empty, real) block the volume symlink — the trueslim
                # canary 7/15 listed unet `[]` exactly here. Empty real dir -> replace with link.
                if os.path.isdir(d) and not os.path.islink(d) and not os.listdir(d):
                    os.rmdir(d)
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
            hs = _hoststore(repo, rel)
            leaf = os.path.join(MODELS_DIR, *parts)
            if hs:
                if os.path.lexists(leaf):
                    os.unlink(leaf)
                os.symlink(hs, leaf)
            _src[rel] = ("HOST-NVMe " + hs) if hs else "volume"
            log(f"model {rel} <- {_src[rel] if hs else 'volume (cache miss)'}")
        _beacon("model_sources", sources=_src)
        _MODEL_SOURCES.update(_src)
        # staged preshards (2026-07-10): if the repo carries fsdp_shards_pr/ and RunPod staged it,
        # point the b2 loader at the host copy (12.4 GB/s measured vs ~0.4 GB/s volume). Loads only;
        # saves still target the volume. Env inherits handler -> comfy -> ray workers (the same
        # chain FSDP_SHARD_DIR already rides). Unset = loader behaviour unchanged.
        try:
            _mangled = "models--" + CACHE_REPO.replace("/", "--")
            _sh = (glob.glob(f"/runpod/model-store/huggingface/{CACHE_REPO}/*/snapshots/*/fsdp_shards_pr")
                   # documented convention (docs.runpod.io model-caching): hf-cache layout on the
                   # volume mount path, served host-locally when the endpoint has a cached model
                   or glob.glob(f"/runpod-volume/huggingface-cache/hub/{_mangled}/snapshots/*/fsdp_shards_pr"))
            if _sh:
                os.environ["HOSTSTORE_SHARDS_DIR"] = _sh[0]
                log("staged preshards -> HOSTSTORE_SHARDS_DIR =", _sh[0])
                _beacon("hoststore_shards", dir=_sh[0], entries=sorted(os.listdir(_sh[0])))
        except Exception as _e:
            log("staged-preshards probe failed:", repr(_e))
    except Exception as e:
        log("setup_models failed:", repr(e))
    _beacon("setup_models_done", dur_s=round(time.time() - _sm_t0, 2))

_setup_models()
_bind("input",  INPUTS_DIR)

def _boot_warmup():
    """Init-preload (unbilled): after ComfyUI is up, run a tiny render so the 14B + FSDP shards +
    CLIP + VAE are resident BEFORE the first billed request (clear_vram=False keeps them loaded).
    First real render then starts warm-equivalent. BOOT_WARMUP=0 disables. Never raises."""
    try:
        if os.environ.get("BOOT_WARMUP", "1") != "1":
            return
        if not _ensure_comfy():
            return
        if max(_vram_used()) > 4000:   # FlashBoot revival with model already resident — skip
            _beacon("warmup_skipped_resident"); return
        # Lever-4 (7/12): warm-render at the legal MINIMUM — Wan needs 4k+1 frames, so 5 is the
        # floor (was 9). Env-overridable so rollback is a template env edit, not a redeploy.
        # A true load-only warmup was investigated and rejected: the FSDP materialization is
        # welded inside the sampler pass (ray_sample), so skipping the sample skips the load.
        job = {"prompt": "warmup: empty beach, no people",
               "src_video": "workflowtest_pre83_720x1280_driving.mp4",
               "src_mask": "workflowtest_pre83_720x1280_mask_inverted.mp4",
               "src_ref_images": "workflowtest_qwen_beach_realistic_startimage.png",
               "width": int(os.environ.get("WARMUP_RES", "128")),
               "height": int(os.environ.get("WARMUP_RES", "128")),
               "frame_num": int(os.environ.get("WARMUP_FRAMES", "5")),
               "sample_steps": int(os.environ.get("WARMUP_STEPS", "1")),
               "sampler_name": "euler", "scheduler": "beta", "cfg": 1.0}
        pid = None
        for attempt in ("min", "fallback"):
            wf, meta = _build_wf(job, _n_gpus())
            wf["18"]["inputs"]["filename_prefix"] = f"WARMUP/{WORKER_ID}"
            t0 = time.time()
            _beacon("warmup_start", attempt=attempt,
                    **{k: meta[k] for k in ("length", "steps", "width", "height")})
            _hwtele("phase", "warmup_start")
            body = json.dumps({"prompt": wf, "client_id": f"warmup_{WORKER_ID}"}).encode()
            req = urllib.request.Request(URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
            try:
                pid = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["prompt_id"]
                break
            except urllib.error.HTTPError as he:
                _beacon("warmup_validation_failed", attempt=attempt, code=he.code,
                        detail=he.read().decode("utf-8", "replace")[:1500])
                if attempt == "fallback":
                    return                       # known-good also failed
                # graph rejected the min attempt -> retry once with the pre-lever-4 known-good
                # (retry unconditionally: fixes ANY env misconfig incl. steps, W2; an identical
                # retry in the min==known-good case costs one validation round-trip, nothing more)
                job.update(width=128, height=128, frame_num=9, sample_steps=1)
        if pid is None:
            return
        while time.time() - t0 < 900:
            time.sleep(5)
            try: h = _get(f"/history/{pid}")
            except Exception: continue
            if pid in h:
                ok = h[pid].get("status", {}).get("status_str") == "success"
                _beacon("warmup_done", ok=ok, dur_s=round(time.time() - t0, 1), vram=_vram_used())
                _hwtele("phase", f"warmup_done:ok={ok}")
                return
        _beacon("warmup_timeout", dur_s=round(time.time() - t0, 1))
    except Exception as e:
        _beacon("warmup_error", err=str(e)[:200])

# --- teardown logger: stamp the exact moment RunPod kills this process (SIGTERM / atexit), so
# the job-end -> teardown delay becomes measurable instead of gap-bounded. One JSON line appended
# to the volume; wholly fail-safe — can never affect boot or serving.
def _install_teardown_logger():
    try:
        import signal, atexit
        _boot_t = time.time()
        def _stamp(reason):
            try:
                with open(os.path.join(VOL, "teardown_log.txt"), "a") as f:
                    f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": round(time.time(), 1),
                                        "worker": WORKER_ID, "reason": reason,
                                        "uptime_s": round(time.time() - _boot_t, 1)}) + "\n")
            except Exception:
                pass
        atexit.register(lambda: _stamp("atexit"))
        def _on_signal(signum, frame):
            _stamp(f"signal_{signum}")
            raise SystemExit(0)
        for _sig in (signal.SIGTERM, signal.SIGINT):
            try: signal.signal(_sig, _on_signal)
            except Exception: pass
    except Exception:
        pass
_install_teardown_logger()

# --- raylight patch applier: install the baked raylight_nodes_r1.py over raylight's nodes.py
# BEFORE ComfyUI boots. Source is the IMAGE now (hotpatch retired 8/24), same gating: absent
# file = no-op; fail-safe. GATED on RAYLIGHT_PRESTART=1 (2026-07-09): with
# the flag off the worker runs STOCK raylight — flag=0 is a true rollback, no wrapper riding along.
try:
    _rlp_src = "/opt/raylight_nodes_r1.py"
    _rlp_dst = os.path.join(COMFY_DIR, "custom_nodes", "raylight", "src", "raylight", "nodes.py")
    if os.environ.get("RAYLIGHT_PRESTART", "0") == "1" \
            and os.path.isfile(_rlp_src) and os.path.isdir(os.path.dirname(_rlp_dst)):
        shutil.copyfile(_rlp_src, _rlp_dst)
        log(f"raylight nodes patch applied from {_rlp_src}")
except Exception as _e:
    log(f"raylight nodes patch skipped: {_e!r}")

# --- shard cache prewarm: pull the newest preshard files into the OS page cache during the
# unbilled boot window, in parallel with ComfyUI boot + Ray init, so the loader's parallel_read
# (~78 s in) hits RAM instead of the network volume. Read-only and wholly fail-safe; costs zero
# billed time. SHARD_CACHE_PREWARM=0 disables.
_shard_prewarm_state = {"status": "off"}
def _shard_cache_prewarm():
    try:
        if os.environ.get("SHARD_CACHE_PREWARM", "1") != "1":
            return
        sdir = os.environ.get("FSDP_SHARD_DIR", os.path.join(VOL, "fsdp_shards_pr"))
        # PREFERRED: the EXACT shard files the loader used last boot (it writes one manifest line
        # per rank after a successful load). The config version hash is stable across boots, so
        # last boot's files == this boot's files. This replaces the old newest-per-basename glob,
        # which warmed 23 GB of STALE shard versions while the loader read the real ~16 GB cold
        # off the network volume EVERY boot (the 7/08 dead-prewarm bug — parallel_read never <8.6s).
        # File discovery moved verbatim into _prewarm_fileset (shared with the _prewarm_probe
        # job hook). Manifest-preferred for the same reason as before: the newest-per-basename
        # glob once warmed 23 GB of STALE shard versions (the 7/08 dead-prewarm bug).
        files, src = _prewarm_fileset()
        _shard_prewarm_state["src"] = src
        if not files:
            _shard_prewarm_state.update(status="no_files", dir=sdir); return
        nthreads = int(os.environ.get("SHARD_CACHE_PREWARM_THREADS", "8"))
        _shard_prewarm_state.update(status="running", files=len(files))
        t0 = time.time()
        # NEED-ORDERED + shard-interleaved queue (see _prewarm_jobs docstring: the old
        # shards-first order self-defeated under memory pressure — read-first = evicted-first).
        jobs, total = _prewarm_jobs(files)
        # ADAPTIVE (2026-07-11): thread count is per-host (see _adaptive_chunk_read docstring).
        # PREWARM_AUTOTUNE=0, an autotune error, OR a short read (engine swallows per-chunk
        # errors — "no exception" is NOT "success") -> the original fixed pool, whose first
        # read error still propagates to status=error like the pre-autotune code. A STALLED
        # result (wedged mount) skips the fixed retry — it would hang on the same mount.
        res = None
        if os.environ.get("PREWARM_AUTOTUNE", "1") == "1":
            try:
                res = _adaptive_chunk_read(jobs, start_workers=nthreads)
                if res.get("stalled"):
                    gbr = round(res["bytes"] / 2**30, 2)
                    _shard_prewarm_state.update(status=f"stalled@{gbr}gb", gb_read=gbr)
                    log(f"shard_prewarm STALLED at {gbr} GiB of {round(total/2**30,2)} "
                        f"(frozen mount?) — no fixed-pool retry")
                    _beacon("shard_prewarm", stalled=True, gb_read=gbr,
                            planned_gb=round(total / 2**30, 2), tune_trace=res["trace"])
                    return
                if res["bytes"] < 0.90 * total:
                    log(f"shard_prewarm autotune short-read {res['bytes'] >> 20}/{total >> 20} MiB "
                        f"(errors={res['errors']}) -> fixed pool retry")
                    res = None
            except Exception as _ae:
                log(f"prewarm autotune failed -> fixed pool: {_ae!r}")
                res = None
        if res is None:
            def _rd(j):
                p, off, ln = j
                with open(p, "rb", buffering=0) as f:
                    f.seek(off)
                    left = ln
                    while left > 0:
                        b = f.read(min(8 * 1024 * 1024, left))
                        if not b: break
                        left -= len(b)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as ex:
                list(ex.map(_rd, jobs))
            dur = round(time.time() - t0, 1)
            _shard_prewarm_state.update(status="done", gb=round(total / 2**30, 2), dur_s=dur, threads=nthreads)
            log(f"shard_prewarm done {total >> 20} MiB in {dur}s (threads={nthreads}, files={len(files)})")
            _beacon("shard_prewarm", gb=round(total / 2**30, 2), dur_s=dur)
        else:
            dur = round(time.time() - t0, 1)
            gb_read = round(res["bytes"] / 2**30, 2)          # ACTUAL bytes, not the plan (F1)
            _shard_prewarm_state.update(status="done", gb=gb_read, planned_gb=round(total / 2**30, 2),
                                        dur_s=dur, threads=res["chosen_w"], autotune=True)
            log(f"shard_prewarm done {res['bytes'] >> 20}/{total >> 20} MiB in {dur}s "
                f"(autotune chosen_w={res['chosen_w']} avg={res['MBps_avg']} MB/s, "
                f"files={len(files)}, errors={res['errors']})")
            _beacon("shard_prewarm", gb=gb_read, planned_gb=round(total / 2**30, 2), dur_s=dur,
                    chosen_w=res["chosen_w"], MBps_avg=res["MBps_avg"],
                    tail_unmeasured=res["tail_unmeasured"], tune_trace=res["trace"],
                    errors=res["errors"])
    except Exception as e:
        _shard_prewarm_state.update(status=f"error:{str(e)[:80]}")
threading.Thread(target=_shard_cache_prewarm, daemon=True).start()

# --- module load: register the worker HEALTHY first, then warm ComfyUI in the background ---
log(f"boot worker={WORKER_ID} epoch={MODULE_EPOCH} tele={WDIR} on_volume={VOL_WRITABLE} "
    f"vol_exists={os.path.isdir(VOL)} comfy_dir_exists={os.path.isdir(COMFY_DIR)}")
_beacon("boot", on_volume=VOL_WRITABLE, vol_exists=os.path.isdir(VOL),
        pyimport_lag_s=(round(_BOOT_T - _CONTAINER_T0, 2) if _CONTAINER_T0 else None),
        **_cpu_identity())


def _boot_health():
    """Degradation index (<1s healthy): the kernel/FUSE/driver round-trip latencies that
    stretched 2-24x on the 7/14 degraded box while pure-CPU work stayed flat. One beacon per
    boot; the FSDP checksum dur_s is the GPU-sync counterpart (parent stays off-GPU by design)."""
    h = {}
    try:
        t0 = time.time(); subprocess.run(["/bin/sh", "-c", ":"], timeout=20)
        h["spawn_ms"] = round((time.time() - t0) * 1000, 1)
    except Exception:
        h["spawn_ms"] = None
    try:
        t0 = time.time(); subprocess.run(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                                         capture_output=True, timeout=30)
        h["smi_ms"] = round((time.time() - t0) * 1000, 1)
    except Exception:
        h["smi_ms"] = None
    try:
        p = os.path.join(WDIR, ".health_probe")
        t0 = time.time()
        with open(p, "w") as f:
            f.write("x"); f.flush(); os.fsync(f.fileno())
        h["vol_write_ms"] = round((time.time() - t0) * 1000, 1)
        t0 = time.time(); open(p).read()
        h["vol_read_ms"] = round((time.time() - t0) * 1000, 1)
    except Exception:
        h["vol_write_ms"] = h.get("vol_write_ms"); h["vol_read_ms"] = None
    try:  # single-thread MEMORY bench — the 7/15 instrumented boot showed the episode class is
          # memory-side (8 GB alloc 14.4 s + memcpy 15.5 s while the NVMe read took 0.59 s), which
          # vol-RTT probes DON'T catch. 256 MB zero-fill + copy; healthy ≈ tens of ms each.
        t0 = time.time(); _mb = bytearray(256 * 1024 * 1024)
        h["mem_alloc_ms"] = round((time.time() - t0) * 1000, 1)
        t0 = time.time(); _mb2 = bytes(_mb)
        h["mem_copy_ms"] = round((time.time() - t0) * 1000, 1)
        del _mb, _mb2
    except Exception:
        pass
    try:  # host core-frequency snapshot (min/med/max MHz) — the per-core-speed rival's fingerprint
        freqs = []
        base = "/sys/devices/system/cpu"
        for cn in os.listdir(base):
            if cn.startswith("cpu") and cn[3:].isdigit():
                try:
                    with open(f"{base}/{cn}/cpufreq/scaling_cur_freq") as f:
                        freqs.append(int(f.read().strip()) // 1000)
                except Exception:
                    pass
        if not freqs:  # cpufreq sysfs absent in many containers -> /proc/cpuinfo
            freqs = [int(float(l.split(":")[1])) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
        if freqs:
            freqs.sort()
            h["mhz_min"], h["mhz_med"], h["mhz_max"] = freqs[0], freqs[len(freqs)//2], freqs[-1]
    except Exception:
        pass
    _beacon("boot_health", **h)
    return h


def _maybe_recycle(h):
    """DETECT-AND-RECYCLE (2026-07-15): when boot_health reads the degraded-host fingerprint at
    +3s (vol round-trips ~3x healthy: 7/14 boot 1iocdjmshsyx2f read 29.8/50.8ms vs 10.6/26.4
    baseline and went on to a 118s models-resident), exit BEFORE any expensive work so RunPod
    relaunches (~25s cycle, 1-2c) rather than eating a 60-120s boot. Guard rails, in order:
    endpoint-gated (stagingtest only until validated; RECYCLE_ENDPOINTS extends), double-probe
    (a one-off spike must confirm on a 1s-later re-read), and a volume-backed budget of 2
    recycles/worker/hour so a long episode boots through instead of crash-looping (the CA-2
    lesson: unbounded relaunch loops bill forever)."""
    eps = set(e for e in os.environ.get("RECYCLE_ENDPOINTS", "69qtffutk83l3o").split(",") if e)
    if ENDPOINT not in eps:
        return
    thr_r = float(os.environ.get("BOOT_MAX_VOLREAD_MS", "20"))
    thr_w = float(os.environ.get("BOOT_MAX_VOLWRITE_MS", "40"))
    def _trips(d):
        return (d.get("vol_read_ms") or 0) >= thr_r and (d.get("vol_write_ms") or 0) >= thr_w
    if not _trips(h):
        return
    time.sleep(1.0)
    h2 = _boot_health()
    if not _trips(h2):
        log("recycle: first probe tripped, re-probe clean — proceeding"); return
    ledger = os.path.join(WDIR, "recycle_log")
    try:
        now = time.time()
        past = [float(x) for x in open(ledger).read().split()] if os.path.exists(ledger) else []
        recent = [t for t in past if now - t < 3600]
        if len(recent) >= 2:
            log(f"recycle: budget exhausted ({len(recent)} in last hour) — booting through despite {h2}")
            _beacon("boot_recycle_skipped", reason="budget", h=h, h2=h2)
            return
        with open(ledger, "w") as f:
            f.write(" ".join(str(t) for t in recent + [now]))
    except Exception as e:
        log("recycle: ledger error, booting through:", repr(e)); return
    log(f"RECYCLE: degraded boot_health confirmed twice ({h} / {h2}) — exiting for relaunch")
    _beacon("boot_recycle", h=h, h2=h2)
    os._exit(43)


try:
    _maybe_recycle(_boot_health())
except Exception:
    pass


def _ray_head_prestart():
    """Start a standalone Ray head at boot so the ComfyUI process CONNECTS (~1-2s) instead of
    building a cluster (~6s, on the warmup critical path when Comfy comes up fast). Marker file
    gates the connect rewrite in raylight_nodes_r1; no marker = today's behavior exactly.
    Stagingtest-gated until a validation boot passes; RAY_HEAD_PRESTART=0/1 overrides."""
    try:
        import shutil as _sh
        if _sh.which("ray") is None:
            log("ray-head: CLI not found; skip"); return
        # explicit GPU count: the head's raylet inventories resources from ITS env; if the handler's
        # differs from comfy's, workers see wrong GPUs — one candidate mechanism for 7/14's loop
        try:
            _idx = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout.split()
            _ngpu = len(_idx)
        except Exception:
            _ngpu = 0
        if _ngpu < 1:
            log("ray-head: could not count GPUs; skip (connect path stays off)"); return
        # THE 7/14 ROOT CAUSE (proven by the 7/16 stderr beacon): the handler runs with
        # CUDA_VISIBLE_DEVICES="" (parent off-GPU by design), so a head inheriting this env is a
        # 0-GPU cluster -> comfy connects -> actor spawn hangs -> crash-loop. Give the head its
        # own env with the GPUs visible; the handler process itself stays off-GPU.
        _env = dict(os.environ); _env["CUDA_VISIBLE_DEVICES"] = ",".join(_idx)
        t0 = time.time()
        r = subprocess.run(["ray", "start", "--head", "--disable-usage-stats",
                            "--include-dashboard=false", f"--num-gpus={_ngpu}"],
                           capture_output=True, text=True, timeout=120, env=_env)
        if r.returncode == 0:
            with open("/tmp/.ray_head_up", "w") as f:
                f.write(str(time.time()))
            log(f"ray-head: up in {time.time()-t0:.1f}s num_gpus={_ngpu}")
            _beacon("ray_head_up", dur_s=round(time.time() - t0, 1), num_gpus=_ngpu)
        else:
            log(f"ray-head: start FAILED rc={r.returncode} {(r.stderr or '')[-300:]}")
            _beacon("ray_head_fail", rc=r.returncode,
                    stderr=(r.stderr or "")[-400:], stdout=(r.stdout or "")[-200:])
    except Exception as e:
        log("ray-head: exception", repr(e))


# ROUND 2 (2026-07-15, redesigned after the 7/14 breakage): stagingtest-gated by default (prod
# boots exactly as today even with this deployed); RAY_HEAD_PRESTART=0/1 still overrides both ways.
# Hardening vs 7/14: explicit --num-gpus on the head; wrapper is ONE-SHOT (a single connect attempt
# per process, marker deleted after) and VALIDATES cluster resources before accepting the connect.
_RH_DEFAULT = "1" if ENDPOINT in set(
    e for e in os.environ.get("RAY_HEAD_ENDPOINTS", "69qtffutk83l3o,8vgr0kn4br5vy5").split(",") if e) else "0"
if os.environ.get("RAY_HEAD_PRESTART", _RH_DEFAULT) == "1":
    threading.Thread(target=_ray_head_prestart, daemon=True).start()


def _code_fingerprints():
    """Per-boot SHA of the small active code files (audit: A/B runs must prove WHICH hotpatch
    generation ran). Model payload is NOT hashed — fingerprint-verified separately."""
    import hashlib
    fps = {"handler": None, "workflow": None}
    def _sha12(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
    try: fps["handler"] = _sha12(os.path.abspath(__file__))
    except Exception: pass
    try: fps["workflow"] = _sha12(WF_PATH)
    except Exception: pass
    for tag, pat in [("raylight_nodes", f"{COMFY_DIR}/custom_nodes/raylight/**/nodes*.py"),
                     ("model_patcher", f"{COMFY_DIR}/custom_nodes/raylight/**/model_patcher*.py"),
                     ("fsdp_utils", f"{COMFY_DIR}/custom_nodes/raylight/**/fsdp_utils*.py")]:
        try:
            hits = sorted(glob.glob(pat, recursive=True))
            fps[tag] = {os.path.basename(h): _sha12(h) for h in hits} or "absent"
        except Exception:
            fps[tag] = "err"
    return fps


try:
    _beacon("code_fp", **_code_fingerprints())
except Exception:
    pass
# --- pod_telemetry.sh updater: the baked image ships an older pod_telemetry.sh; copy the enhanced one
# (PSI/vmstat/full-meminfo/allproc-io/loadavg streams) from code_hotpatch over HERE BEFORE telemetry
# starts, so the extra streams actually run. Fail-safe (missing file = keep baked version).
try:
    _pt_src = os.path.join(VOL, "code_hotpatch", "pod_telemetry.sh")
    _pt_dst = os.path.join(HERE, "pod_telemetry.sh")
    if os.path.isfile(_pt_src):
        shutil.copyfile(_pt_src, _pt_dst)
        log("pod_telemetry.sh updated from code_hotpatch")
except Exception as _e:
    log("pod_telemetry.sh update skipped:", repr(_e))

# --- cache-residency logger: DIRECT read of "is each shard cached RIGHT NOW" via mincore, every 1s
# from boot -> residency.csv on the volume. We READ the timeline: prewarm caches rank_N (goes to
# 100%), whether it STAYS or gets evicted, and its EXACT residency when the loader reads it. No
# timing inference. Also logs whole-box Cached + MemAvailable (the shared-host memory-pressure signal).
def _mincore_pct(path):
    """% of file pages resident in page cache via mincore. Returns a float, or an ERROR TAG string
    (never silent None) so a broken instrument is visible in the log, not invisible."""
    import ctypes, mmap as _mm
    mm = None
    try:
        sz = os.path.getsize(path)
        if sz <= 0: return "sz0"
        fd = os.open(path, os.O_RDONLY)
        try:
            mm = _mm.mmap(fd, sz, access=_mm.ACCESS_COPY)   # COW -> writable mapping (from_buffer needs writable), file untouched
        finally:
            os.close(fd)
        ps = os.sysconf("SC_PAGE_SIZE"); n = (sz + ps - 1) // ps
        vec = (ctypes.c_ubyte * n)()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        buf = (ctypes.c_char * sz).from_buffer(mm)
        rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(sz), vec)
        if rc != 0:
            res = "mc_errno%d" % ctypes.get_errno()
        else:
            res = round(100.0 * bytes(vec).count(1) / n, 1)   # mincore sets bit0=resident; count(1) is C-fast
        buf = None   # release the ctypes export before closing the mmap
        return res
    except Exception as e:
        return "ex:" + type(e).__name__
    finally:
        try:
            if mm is not None: mm.close()
        except Exception:
            pass
def _residency_logger():
    try:
        files = []
        for mp in sorted(glob.glob(os.path.join(VOL, "fsdp_shards_pr", ".prewarm_manifest_rank*"))):
            try:
                p = open(mp).read().strip()
                if os.path.isfile(p): files.append(p)
            except Exception: pass
        for rel in ("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "vae/wan_2.1_vae.safetensors",
                    "loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors"):
            p = os.path.realpath(os.path.join(VOL_MODELS, rel))
            if os.path.isfile(p): files.append(p)
        csv = os.path.join(WDIR, "residency.csv")
        with open(csv, "w") as f:
            f.write("ts,boot_s,cached_gib,memavail_gib," + ",".join(os.path.basename(x) for x in files) + "\n")
        while True:
            cg = ma = 0.0
            try:
                for l in open("/proc/meminfo"):
                    if l.startswith("Cached:"): cg = int(l.split()[1])/(1024*1024)
                    elif l.startswith("MemAvailable:"): ma = int(l.split()[1])/(1024*1024)
            except Exception: pass
            row = [f"{time.time():.3f}", f"{time.time()-_BOOT_T:.1f}", f"{cg:.1f}", f"{ma:.1f}"] + \
                  [str(_mincore_pct(x)) for x in files]
            with open(csv, "a") as f: f.write(",".join(row) + "\n")
            time.sleep(1)
    except Exception as e:
        log("residency_logger failed:", repr(e))
threading.Thread(target=_residency_logger, daemon=True).start()

# --- COMPLETE per-process/thread sampler (proc_sampler.py from code_hotpatch): every kernel observable
# for every thread of every process, from boot -> WDIR/full/. Niced SEPARATE process (no GIL contention
# with the handler). This is the "reconstruct any process's full behaviour from the logs" capture.
try:
    _ps_src = os.path.join(VOL, "code_hotpatch", "proc_sampler.py")
    _ps_dst = os.path.join(HERE, "proc_sampler.py")
    if os.path.isfile(_ps_src):
        shutil.copyfile(_ps_src, _ps_dst)
        os.makedirs(os.path.join(WDIR, "full"), exist_ok=True)
        subprocess.Popen(["nice", "-n", "19", sys.executable, _ps_dst, os.path.join(WDIR, "full"), "1"],
                         stdout=subprocess.DEVNULL, stderr=open(os.path.join(WDIR, "proc_sampler.err"), "w"))
        log("proc_sampler launched -> WDIR/full/")
    else:
        log("proc_sampler.py not on volume — skipped")
except Exception as _e:
    log("proc_sampler launch skipped:", repr(_e))

# preserve the PREVIOUS boot's hw telemetry before 'start' truncates the streams — same rationale
# as comfy_prev: a killed container's last-seconds hw signals (rate collapse vs OOM ramp vs GPU
# stall) are the only crash-mechanism evidence. Prune to the 6 newest so the volume doesn't grow.
try:
    _hw_dir = os.path.join(WDIR, "hw")
    if os.path.isdir(_hw_dir) and os.listdir(_hw_dir):
        os.rename(_hw_dir, os.path.join(WDIR, f"hw_prev_{int(time.time())}"))
    _prev = sorted(d for d in os.listdir(WDIR) if d.startswith("hw_prev_"))
    for _d in _prev[:-6]:
        shutil.rmtree(os.path.join(WDIR, _d), ignore_errors=True)
except Exception as _e:
    log("hw_prev preservation skipped:", repr(_e))

_hwtele("start")   # full hw telemetry running BEFORE ComfyUI launches → captures the cold load (read vs dequant)
threading.Thread(target=_boot_warmup, daemon=True).start()   # ensures comfy + preloads model; NEVER blocks start()
runpod.serverless.start({"handler": handler})
