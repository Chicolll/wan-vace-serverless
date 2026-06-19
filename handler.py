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
VOL      = "/runpod-volume/native-xdit"           # network volume mount: INPUTS only now (model is baked in image)
MODEL    = "/opt/xdit/model"                       # 75GB model BAKED into the image at build time (see Dockerfile):
                                                  # no runtime download, no cache lookup, no write-disk needed.


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
    if not os.path.exists(f"{MODEL}/config.json"):
        return {"error": f"baked model missing at {MODEL} — check the image build"}

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
