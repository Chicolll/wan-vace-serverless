# Serverless worker image — Raylight FSDP, SLIM build.
#
# Same env as the 15 GB image (torch 2.8 cu126, pinned transformers/diffusers,
# Raylight ec3ac78 + patches) minus the waste: the old image was built on the
# CUDA -devel base with torch 2.4, then force-reinstalled torch 2.8 on top —
# carrying two full PyTorch stacks and a CUDA toolkit nothing uses at runtime.
# This one: minimal Python base, torch installed exactly once. The pip cu126
# wheels bundle all CUDA runtime libraries; the GPU driver comes from the host.
#
# Measured motivation: cold-start probe 2026-07-03 — image pull + container
# start is 585.5 s of the ~609 s delay (96%), at ~26 MB/s for 15 GB.
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 COMFY_DIR=/opt/ComfyUI
WORKDIR /opt

# System deps: ffmpeg (VHS video encode), git (clone at build), libgl1+libglib2.0-0 (opencv
# imports in node packs). python:3.11-slim puts python at /usr/local/bin; the install script
# and the old image use /usr/bin/python3.11 — symlink so both paths work.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11

# ComfyUI + custom nodes (Raylight pinned to proven commit) — same as the fat image.
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && cd /opt/ComfyUI/custom_nodes \
 && git clone https://github.com/komikndr/raylight && (cd raylight && git checkout ec3ac78) \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
 && git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF

# Torchaudio ABI patch (LTX audio nodes not needed for VACE) — unchanged.
COPY patch_raylight_init.py /opt/patch_raylight_init.py
RUN python3.11 /opt/patch_raylight_init.py

# Python env — same recipe and ordering as raylight_full_install.sh (torch 2.8 LAST,
# after ComfyUI reqs, so nothing downgrades it), but no prior torch to fight: the
# --force-reinstall is unnecessary here and omitted so no duplicate stack exists.
COPY raylight_full_install.sh /opt/raylight_full_install.sh
RUN COMFY_DIR=/opt/ComfyUI bash /opt/raylight_full_install.sh && tail -n 60 /root/install.log

# Modified Raylight FSDP files with pre-shard support — unchanged.
COPY model_patcher_remote.py /opt/ComfyUI/custom_nodes/raylight/model_patcher_remote.py
COPY fsdp_utils_remote.py    /opt/ComfyUI/custom_nodes/raylight/fsdp_utils_remote.py

# Hard build-time import gate — unchanged.
RUN python3.11 -c "\
import torch, ray, diffusers, transformers, runpod; \
from torch.distributed.fsdp import fully_shard; \
import inspect; assert 'ignored_params' in inspect.signature(fully_shard).parameters; \
print('IMG_ENV_OK torch', torch.__version__, 'diffusers', diffusers.__version__, \
      'transformers', transformers.__version__, 'FSDP2 ok')"

# Handler + launcher + workflow + model-path map — unchanged.
COPY handler_raylight.py    /opt/handler_raylight.py
COPY comfy_launch.py        /opt/comfy_launch.py
COPY pod_telemetry.sh       /opt/pod_telemetry.sh
COPY raylight_vace_wf.json  /opt/raylight_vace_wf.json
COPY extra_model_paths.yaml /opt/extra_model_paths.yaml

ENV WF_PATH=/opt/raylight_vace_wf.json EXTRA_MODEL_PATHS=/opt/extra_model_paths.yaml
ENV FSDP_SHARD_DIR=/runpod-volume/fsdp_shards
CMD ["python3.11", "-u", "/opt/handler_raylight.py"]
