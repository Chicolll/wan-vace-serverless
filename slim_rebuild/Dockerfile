# TRUE-SLIM rebuild (2026-07-14) — replaces ghcr.io/chicolll/wan-vace-fsdp:slim-baked-b4cc985
# (14.86 GiB compressed: RunPod pod base with CUDA-devel, Jupyter, nginx, SSH, Slurm, 5 Pythons,
# and a DUPLICATE PyTorch — see startup-latency-reaudit-2026-07-13.md). Target: <=8 GiB.
#
# Sources of truth: the LIVE container's pip freeze + ldconfig (captured 7/14, worker
# ohko3dcjfexr5n -> requirements.lock) — NOT the old cu126 Dockerfile. Torch's CUDA userspace
# comes entirely from pip nvidia-*-cu12 wheels, so NO system CUDA toolkit is installed here.
# GATES before any GPU test: (1) manifest <=8 GiB compressed, no multi-GiB duplicate layers
# (docker buildx imagetools inspect); (2) import gate below passes at build time;
# (3) ONE canary boot on stagingtest validating topology/NCCL + workflow before any prod repoint.
FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

# opencv needs libGL/glib; git only for the clone layer (removed after)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI 0.27.0 + node packs, shallow, histories stripped (the fat image shipped a
# non-shallow raylight clone). raylight pinned to the proven ec3ac78.
RUN git clone --depth 1 --branch v0.27.0 https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && cd /opt/ComfyUI/custom_nodes \
 && git clone https://github.com/komikndr/raylight && (cd raylight && git checkout ec3ac78) \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
 && git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF \
 && find /opt/ComfyUI -name .git -type d -prune -exec rm -rf {} + \
 && apt-get purge -y git && apt-get autoremove -y

# torch trio from the cu128 index FIRST (exact live versions; pulls the nvidia-*-cu12 wheel set),
# then the full lock (everything pinned -> no resolver drift, nothing can downgrade torch).
RUN pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128
COPY requirements.lock /tmp/requirements.lock
# --no-deps: the lock IS the complete closure (live pip freeze). The live env is internally
# inconsistent (kernels 0.16.0 declares huggingface-hub>=1.10.0 but live runs 0.36.2), so the
# resolver refuses to reproduce it (first Actions build failed exactly there). The import gate
# below is the real correctness check.
RUN pip install --no-deps -r /tmp/requirements.lock

# raylight init patch (same step the live image ran)
COPY patch_raylight_init.py /opt/patch_raylight_init.py
RUN python /opt/patch_raylight_init.py

# /opt runtime files (hotpatch copies from the volume overwrite these at container start,
# exactly as today — dockerArgs stays unchanged)
COPY comfy_launch.py extra_model_paths.yaml /opt/
COPY handler_raylight_fsdp.py /opt/handler_raylight.py
COPY raylight_vace_wf_unet_fsdp.json /opt/raylight_vace_wf_unet_fsdp.json
ENV WF_PATH=/opt/raylight_vace_wf_unet_fsdp.json EXTRA_MODEL_PATHS=/opt/extra_model_paths.yaml \
    COMFY_DIR=/opt/ComfyUI

# import gate: fails the BUILD if the runtime closure is broken (cheaper than a GPU canary)
RUN python - <<'EOF'
import torch, ray, diffusers, transformers, xfuser, einops, safetensors, av, cv2
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.version.cuda == "12.8", torch.version.cuda
from torch.distributed.fsdp import fully_shard
import inspect
assert "ignored_params" in inspect.signature(fully_shard).parameters
print("IMPORT_GATE_OK")
EOF

# bake: bytecode + matplotlib font cache (the Dockerfile.bake wins, folded in)
RUN python -m compileall -q -j0 /opt/ComfyUI || true
RUN HOME=/root python -c "import matplotlib.font_manager as fm; fm.fontManager; print('fonts cached')"

CMD ["python", "-u", "/opt/handler_raylight.py"]
