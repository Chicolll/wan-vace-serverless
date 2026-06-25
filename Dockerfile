# Serverless worker image — Raylight (ComfyUI + Wan2.1-VACE-14B Q4 GGUF + FusionX) USP render.
#
# Mirrors the native lean-image pattern: bakes env + ComfyUI + the custom nodes the workflow uses (~18GB). The ~18GB
# MODEL set is NOT baked — it comes from the mounted network volume at runtime (baking it made a ~37GB image this
# serverless endpoint can't place a worker for; 2026-06-25). Reconstructs the PROVEN pod recipe (raylight_full_install.sh:
# torch 2.6 + ray + xfuser + node reqs + pins transformers 4.49 / diffusers 0.33 / nccl 2.28.9 / numpy<2) into
# SYSTEM python3.11. Build for linux/amd64 (RunPod arch).
#
# CONFIRM BEFORE A PRODUCTION BUILD (flagged, not guessed):
#  - custom-node REVISIONS: Raylight is pinned to ec3ac78 (the proven rev); the others clone latest. Pin them to the
#    volume's working custom_nodes revs for exact reproducibility. The set below is scoped to nodes raylight_vace_wf.json
#    actually references (WanVaceToVideo is CORE ComfyUI). If a "missing node" error appears, add WanVideoWrapper /
#    WanVaceAdvanced / Easy-Use (the broader recipe set).
#  - MODEL PATHS: extra_model_paths.yaml points at the volume's ComfyUI model tree — verify the subpaths (esp. where
#    the GGUF lives) against the actual volume layout.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 COMFY_DIR=/opt/ComfyUI
WORKDIR /opt

# ComfyUI + only the custom nodes raylight_vace_wf.json references (Raylight pinned; canonical repos).
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && cd /opt/ComfyUI/custom_nodes \
 && git clone https://github.com/komikndr/raylight && (cd raylight && git checkout ec3ac78) \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# Proven Raylight env recipe into SYSTEM python3.11, against /opt/ComfyUI (COMFY_DIR override).
COPY raylight_full_install.sh /opt/raylight_full_install.sh
RUN COMFY_DIR=/opt/ComfyUI bash /opt/raylight_full_install.sh; tail -n 60 /root/install.log || true

# The serverless handler needs the `runpod` python SDK — NOT in the Raylight recipe (the native image installed it
# separately). Missing it = `import runpod` crashes every worker on boot before it can register. Install it here.
RUN python3.11 -m pip install runpod

# Hard build-time import gate — a broken dep (INCLUDING runpod) fails the BUILD, never a live (metered) worker.
RUN python3.11 -c "import torch, xfuser, ray, diffusers, transformers, yunchang, runpod; \
print('IMG_ENV_OK', torch.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"

# NOTE (2026-06-25): baking the ~19GB model set into the image made it ~37GB, which this serverless endpoint
# CANNOT place a worker for (confirmed: 0 workers ever booted across 2 attempts, container-disk bump 20→50 no help,
# console Logs show no worker entry at all — a placement/size limit, not the handler code). So models are NOT baked;
# they come from the volume, and the GGUF gets fast-loaded via RunPod's host-model-cache (set the endpoint Model
# field to mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF). handler_raylight._setup_models() prefers the host-cache
# path and falls back to the volume.

# Handler + launcher + workflow + model-path map (models themselves mount from the volume at runtime).
COPY handler_raylight.py   /opt/handler_raylight.py
COPY comfy_launch.py       /opt/comfy_launch.py
COPY pod_telemetry.sh      /opt/pod_telemetry.sh
COPY raylight_vace_wf.json /opt/raylight_vace_wf.json
COPY extra_model_paths.yaml /opt/extra_model_paths.yaml

ENV WF_PATH=/opt/raylight_vace_wf.json EXTRA_MODEL_PATHS=/opt/extra_model_paths.yaml
CMD ["python3.11", "-u", "/opt/handler_raylight.py"]
