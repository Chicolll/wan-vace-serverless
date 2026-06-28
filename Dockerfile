# Serverless worker image — Raylight FSDP (2-GPU fp8 VACE + USP) render.
#
# Torch 2.8 (required for FSDP fp8 state-dict assertions), pre-shard support.
# Models NOT baked (37GB image = can't place workers). They come from host NVMe
# cache (RunPod model cache) or the network volume. Pre-sharded FSDP checkpoints
# live on the volume (generated on first render, reused on subsequent cold starts).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 COMFY_DIR=/opt/ComfyUI
WORKDIR /opt

# ComfyUI + custom nodes (Raylight pinned to proven commit).
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && cd /opt/ComfyUI/custom_nodes \
 && git clone https://github.com/komikndr/raylight && (cd raylight && git checkout ec3ac78) \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
 && git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF

# Torchaudio ABI patch: wrap nodes_lt and comfyui_ltxv imports in Raylight __init__.py.
# These nodes load torchaudio operators that ABI-mismatch with torch 2.8 on the base
# image (torch 2.4 torchaudio). They're LTX audio/video nodes, not needed for VACE.
COPY patch_raylight_init.py /opt/patch_raylight_init.py
RUN python3.11 /opt/patch_raylight_init.py

# Torch 2.8 env recipe (proven via raylight_build_28.sh on pod).
COPY raylight_full_install.sh /opt/raylight_full_install.sh
RUN COMFY_DIR=/opt/ComfyUI bash /opt/raylight_full_install.sh && tail -n 60 /root/install.log

# COPY modified Raylight FSDP files with pre-shard support.
COPY model_patcher_remote.py /opt/ComfyUI/custom_nodes/raylight/model_patcher_remote.py
COPY fsdp_utils_remote.py    /opt/ComfyUI/custom_nodes/raylight/fsdp_utils_remote.py

# Hard build-time import gate.
RUN python3.11 -c "\
import torch, ray, diffusers, transformers, runpod; \
from torch.distributed.fsdp import fully_shard; \
import inspect; assert 'ignored_params' in inspect.signature(fully_shard).parameters; \
print('IMG_ENV_OK torch', torch.__version__, 'diffusers', diffusers.__version__, \
      'transformers', transformers.__version__, 'FSDP2 ok')"

# Handler + launcher + workflow + model-path map.
COPY handler_raylight.py    /opt/handler_raylight.py
COPY comfy_launch.py        /opt/comfy_launch.py
COPY pod_telemetry.sh       /opt/pod_telemetry.sh
COPY raylight_vace_wf.json  /opt/raylight_vace_wf.json
COPY extra_model_paths.yaml /opt/extra_model_paths.yaml

ENV WF_PATH=/opt/raylight_vace_wf.json EXTRA_MODEL_PATHS=/opt/extra_model_paths.yaml
ENV FSDP_SHARD_DIR=/runpod-volume/fsdp_shards
CMD ["python3.11", "-u", "/opt/handler_raylight.py"]
