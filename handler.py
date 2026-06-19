#!/usr/bin/env python3
"""RunPod Serverless handler for the native Wan2.1-VACE-14B xfuser render.

Wraps the EXACT invocation proven on 2026-06-18 (1xH100 rendered the cat clip end-to-end):
  - 1 GPU  -> plain python + --offload_model True --t5_cpu  (fits 80GB by streaming the model)
  - N GPUs -> torchrun --nproc_per_node N + USP (--dit_fsdp --t5_fsdp --ulysses_size N --ring_size 1);
              activations split across GPUs, so offload is NOT needed and it's faster.

Image bakes venv + repos + deps (see Dockerfile). The 75GB model is NOT baked — it lives on the network
volume, mounted by serverless at /runpod-volume. Request gives input FILENAMES already staged on the volume
under native-xdit/inputs/ (or absolute /runpod-volume paths) + prompt + params. Returns the mp4 as base64.

Request schema (event["input"]):
  prompt           (str, required)
  src_video        (str, required)  filename under native-xdit/inputs/ OR absolute /runpod-volume path
  src_mask         (str, required)
  src_ref_images   (str, required)
  size             (str, default "720*1280")   one of vace-14B's SUPPORTED_SIZES
  frame_num        (int, default 81)           4n+1
  sample_steps     (int, default 10)
  n_gpus           (int, default = all visible GPUs on the worker)
"""
import os, subprocess, base64, time
import runpod

VENV_PY  = "/opt/xdit/venv/bin/python"
TORCHRUN = "/opt/xdit/venv/bin/torchrun"
VACE_DIR = "/opt/xdit/VACE"
WAN_DIR  = "/opt/xdit/Wan2.1"
VOL      = "/runpod-volume/native-xdit"           # network volume mount: INPUTS staged here
HF_REPO  = "Wan-AI/Wan2.1-VACE-14B"


def _find_model():
    """Locate the model dir RunPod's 'Cached model' pre-stage put on disk (no download, no write).
    Prefers the FAST host model-store (/runpod/model-store, host-local) over the slow MooseFS volume cache.
    Confirmed layout (2026-06-19 debug invoke): /runpod/model-store/huggingface/<MODEL_NAME>/<MODEL_REVISION>/."""
    import glob
    mn = (os.environ.get("MODEL_NAME") or "wan-ai/wan2.1-vace-14b").strip("/")
    mr = os.environ.get("MODEL_REVISION") or ""
    cands = []
    # 1) FAST host model-store (preferred) — flat revision dir, then snapshot layout
    if mr:
        cands += [f"/runpod/model-store/huggingface/{mn}/{mr}",
                  f"/runpod/model-store/huggingface/{mn}/{mr}/snapshots/{mr}"]
    cands += glob.glob(f"/runpod/model-store/huggingface/{mn}/**/config.json", recursive=True)
    # 2) volume HF cache (slow MooseFS) — fallback
    cands += glob.glob("/runpod-volume/huggingface-cache/hub/models--*ace-14b/snapshots/*", recursive=True)
    cands += glob.glob("/runpod-volume/huggingface-cache/hub/models--*VACE-14B/snapshots/*", recursive=True)
    # 3) explicit prior copies
    cands += [f"{VOL}/model/Wan2.1-VACE-14B", "/opt/xdit/model"]
    for c in cands:
        d = os.path.dirname(c) if c.endswith("config.json") else c
        if d and os.path.exists(os.path.join(d, "config.json")):
            return d
    return None


def _debug_fs():
    """Report env + filesystem so we can find where RunPod pre-staged the model."""
    import glob
    env = {k: v for k, v in os.environ.items()
           if any(t in k.upper() for t in ("HF", "HUGGING", "CACHE", "MODEL", "RUNPOD", "VOLUME"))}
    found = []
    for base in ["/runpod-volume", "/root/.cache", "/cache", "/runpod", "/opt/xdit", os.path.expanduser("~/.cache")]:
        try:
            found += glob.glob(f"{base}/**/config.json", recursive=True)[:20]
            found += glob.glob(f"{base}/**/*Wan2.1-VACE-14B*", recursive=True)[:20]
        except Exception as e:
            found.append(f"ERR {base}: {e}")
    listings = {}
    for d in ["/runpod-volume", "/root/.cache/huggingface", "/runpod-volume/huggingface-cache"]:
        try:
            listings[d] = os.listdir(d)[:30]
        except Exception as e:
            listings[d] = f"ERR {e}"
    return {"env": env, "config_json_hits": sorted(set(found))[:40], "listings": listings}


def _gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return max(1, sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU ")))
    except Exception:
        return 1


def _resolve(p):
    if not p:
        return p
    return p if p.startswith("/") else f"{VOL}/inputs/{p}"


def handler(event):
    job = (event or {}).get("input", {}) or {}
    if job.get("debug"):
        return _debug_fs()
    prompt = job.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    size   = job.get("size", "720*1280")
    frames = int(job.get("frame_num", 81))
    steps  = int(job.get("sample_steps", 10))
    src_video = _resolve(job.get("src_video"))
    src_mask  = _resolve(job.get("src_mask"))
    src_ref   = _resolve(job.get("src_ref_images"))
    for label, path in (("src_video", src_video), ("src_mask", src_mask), ("src_ref_images", src_ref)):
        if not path or not os.path.exists(path):
            return {"error": f"missing input {label}: {path}"}
    MODEL = _find_model()
    if not MODEL:
        return {"error": "model not found on disk — RunPod cached-model pre-stage path unknown",
                "debug": _debug_fs()}

    n = int(job.get("n_gpus") or _gpu_count())
    out_file = f"/tmp/out_{int(time.time())}.mp4"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{WAN_DIR}:{VACE_DIR}:" + env.get("PYTHONPATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    base = [
        "vace/vace_wan_inference.py", "--model_name", "vace-14B", "--size", size,
        "--frame_num", str(frames), "--ckpt_dir", MODEL,
        "--src_video", src_video, "--src_mask", src_mask, "--src_ref_images", src_ref,
        "--sample_steps", str(steps), "--prompt", prompt, "--save_file", out_file,
    ]
    if n <= 1:
        # PROVEN single-GPU path (offload streams the model; t5 on CPU; fits 80GB at 720x1280).
        cmd = [VENV_PY] + base + ["--offload_model", "True", "--t5_cpu"]
    else:
        # Multi-GPU USP (sequence-parallel). N must divide the 40 attention heads (2/4/8 all do).
        cmd = [TORCHRUN, "--nproc_per_node", str(n)] + base + \
              ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(n), "--ring_size", "1"]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=VACE_DIR, env=env, capture_output=True, text=True)
    dur = round(time.time() - t0, 1)

    if proc.returncode != 0 or not os.path.exists(out_file):
        return {"error": "render failed", "returncode": proc.returncode, "n_gpus": n,
                "stderr": proc.stderr[-4000:], "stdout": proc.stdout[-1500:]}

    with open(out_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    sz = os.path.getsize(out_file)
    try:
        os.remove(out_file)
    except OSError:
        pass
    return {"video_base64": data, "bytes": sz, "seconds": dur, "n_gpus": n,
            "size": size, "frame_num": frames, "sample_steps": steps}


runpod.serverless.start({"handler": handler})
