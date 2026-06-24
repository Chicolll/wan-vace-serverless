#!/bin/bash
# PROVEN Raylight recipe into SYSTEM python3.11 on CONTAINER DISK (volume is quota-full).
# Pins LAST so they win over whatever node reqs pull. Robust: plain per-cmd append to log.
set -u
C="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"   # pod default; image build overrides via COMFY_DIR=/opt/ComfyUI
PY=/usr/bin/python3.11
PIP="$PY -m pip install"
LOG=/root/install.log
: > "$LOG"
echo "=== INSTALL START $(date -u) ===" >> "$LOG"

echo "=== [1] torch 2.6 cu124 ===" >> "$LOG"
$PIP torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 >> "$LOG" 2>&1 && echo "OK_TORCH" >> "$LOG" || echo "FAIL_TORCH" >> "$LOG"

echo "=== [2] Raylight deps ===" >> "$LOG"
$PIP "ray>=2.48.0" "xfuser>=0.4.4" kernels huggingface_hub hf_transfer >> "$LOG" 2>&1 && echo "OK_DEPS" >> "$LOG" || echo "FAIL_DEPS" >> "$LOG"

echo "=== [3] ComfyUI core reqs ===" >> "$LOG"
$PIP -r "$C/requirements.txt" >> "$LOG" 2>&1 && echo "OK_COMFY" >> "$LOG" || echo "FAIL_COMFY" >> "$LOG"

echo "=== [4] node reqs ===" >> "$LOG"
for n in ComfyUI-GGUF ComfyUI-WanVideoWrapper ComfyUI-WanVaceAdvanced ComfyUI-KJNodes ComfyUI-VideoHelperSuite ComfyUI-Easy-Use; do
  R="$C/custom_nodes/$n/requirements.txt"
  if [ -f "$R" ]; then echo "-- $n" >> "$LOG"; $PIP -r "$R" >> "$LOG" 2>&1 || echo "WARN $n" >> "$LOG"; fi
done
echo "OK_NODES" >> "$LOG"

echo "=== [5] PINS LAST (override) ===" >> "$LOG"
$PIP "transformers==4.49.0" "diffusers==0.33.1" >> "$LOG" 2>&1 && echo "OK_PINS" >> "$LOG" || echo "FAIL_PINS" >> "$LOG"
$PIP --force-reinstall --no-deps "nvidia-nccl-cu12==2.28.9" >> "$LOG" 2>&1 && echo "OK_NCCL" >> "$LOG" || echo "FAIL_NCCL" >> "$LOG"
$PIP "numpy<2" >> "$LOG" 2>&1 && echo "OK_NUMPY" >> "$LOG" || echo "FAIL_NUMPY" >> "$LOG"

echo "=== [6] VERIFY ===" >> "$LOG"
$PY -c "import torch,xfuser,ray,yunchang,diffusers,transformers; print('VERIFY torch',torch.__version__,'xfuser ok ray',ray.__version__,'diffusers',diffusers.__version__,'transformers',transformers.__version__)" >> "$LOG" 2>&1 && echo "IMPORTS_OK" >> "$LOG" || echo "IMPORTS_FAIL" >> "$LOG"
echo "=== INSTALL DONE $(date -u) ===" >> "$LOG"
