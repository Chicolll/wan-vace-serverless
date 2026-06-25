# Serverless worker image — Raylight (ComfyUI + Wan2.1-VACE-14B Q4 GGUF + FusionX) USP render.
#
# Mirrors the native lean-image pattern: bakes env + ComfyUI + the custom nodes the workflow uses (~18GB). The 4
# workflow MODELS (~19GB Q4 GGUF set) ARE now baked onto local disk (bake_models.py) — the 85GB wall was the bf16
# native model, NOT this quant; baking kills the per-cold network-volume read. Reconstructs the PROVEN pod recipe (raylight_full_install.sh:
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

# Bake the 4 workflow models onto the image's LOCAL disk (~19GB): no per-cold-start network-volume read — the ~115s
# MooseFS read that dominated cold load is gone (cold = dequant + Ray init only). Sidesteps RunPod's one-cached-model
# limit (4 files / 3 repos). Public HF, sha256-verified; a mismatch FAILS the build. hf_transfer speeds the pull;
# HF_HOME + temp dir are cleaned in-layer so only the baked files remain. The handler finds these local files ahead
# of the volume (no code change needed — _setup_models keeps existing real dirs and only symlinks missing ones).
COPY bake_models.py /opt/bake_models.py
RUN python3.11 -m pip install --no-cache-dir huggingface_hub hf_transfer && \
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/tmp/hfhome COMFY_DIR=/opt/ComfyUI python3.11 /opt/bake_models.py && \
    rm -rf /tmp/hfhome /tmp/bake_dl

# Handler + launcher + workflow + model-path map (models themselves mount from the volume at runtime).
COPY handler_raylight.py   /opt/handler_raylight.py
COPY comfy_launch.py       /opt/comfy_launch.py
COPY pod_telemetry.sh      /opt/pod_telemetry.sh
COPY raylight_vace_wf.json /opt/raylight_vace_wf.json
COPY extra_model_paths.yaml /opt/extra_model_paths.yaml

ENV WF_PATH=/opt/raylight_vace_wf.json EXTRA_MODEL_PATHS=/opt/extra_model_paths.yaml
CMD ["python3.11", "-u", "/opt/handler_raylight.py"]
