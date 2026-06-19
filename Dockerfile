# Serverless worker image for native Wan2.1-VACE-14B xfuser render.
# Bakes venv + repos + deps ONLY (~12-15GB). The 75GB model is NOT baked — it mounts from the network volume
# at /runpod-volume (RunPod recommends against baking large models; image stays lean for fast cold start).
# Reconstructs the EXACT environment validated 2026-06-18 (torch 2.6 + xfuser + flash-attn matched ABI + Wan/VACE
# deps incl ftfy/pycocotools/framework). Build for linux/amd64 (RunPod arch).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
WORKDIR /opt/xdit

# OWN venv with torch 2.6 (image ships 2.4, too old for xfuser's diffusers>=0.33 flash-attn-3 custom-op schema).
RUN python3.11 -m venv /opt/xdit/venv
ENV PATH=/opt/xdit/venv/bin:$PATH

RUN pip install -U pip wheel setuptools \
 && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
 && pip install ninja "xfuser>=0.4.1" \
 && pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
 && pip install ftfy pycocotools easydict opencv-python-headless imageio imageio-ffmpeg scikit-image matplotlib scipy onnxruntime runpod

# Repos + their requirements. ftfy/pycocotools above are REQUIRED — vace_wan_inference.py imports the whole
# annotator stack at module load even for masked inpainting (2026-06-18 burn).
RUN git clone --depth 1 https://github.com/ali-vilab/VACE /opt/xdit/VACE \
 && git clone --depth 1 https://github.com/Wan-Video/Wan2.1 /opt/xdit/Wan2.1 \
 && pip install -r /opt/xdit/Wan2.1/requirements.txt \
 && if [ -f /opt/xdit/VACE/requirements/framework.txt ]; then pip install -r /opt/xdit/VACE/requirements/framework.txt; fi \
 && pip install "numpy<2"   # Wan requires numpy<2; pin LAST so it wins the opencv-headless>=2 conflict

COPY handler.py /opt/xdit/handler.py

# import sanity at BUILD time — a broken dep fails the build, never a live worker.
RUN python -c "import torch, xfuser, flash_attn, ftfy, pycocotools, runpod; print('IMG_ENV_OK', torch.__version__)"

CMD ["python", "-u", "/opt/xdit/handler.py"]
