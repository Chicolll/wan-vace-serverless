#!/bin/bash
# Raylight env recipe for FSDP fp8: PyTorch 2.8 + NCCL 2.28.9 into SYSTEM python3.11.
# Adapted from raylight_build_28.sh (proven on pod) into Dockerfile-friendly form.
# Failures in critical steps (torch, verify) are FATAL and stop the Docker build.
set -euo pipefail
C="${COMFY_DIR:-/opt/ComfyUI}"
PY=/usr/bin/python3.11
PIP="$PY -m pip install"
LOG=/root/install.log
: > "$LOG"
echo "=== INSTALL START $(date -u) ===" | tee -a "$LOG"

echo "=== [1] torch 2.8 cu126 ===" | tee -a "$LOG"
$PIP torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 2>&1 | tee -a "$LOG"

echo "=== [2] Raylight deps ===" | tee -a "$LOG"
$PIP "ray>=2.48.0" "xfuser>=0.4.4" kernels huggingface_hub hf_transfer 2>&1 | tee -a "$LOG"

echo "=== [3] ComfyUI core reqs ===" | tee -a "$LOG"
$PIP -r "$C/requirements.txt" 2>&1 | tee -a "$LOG"

echo "=== [4] node reqs ===" | tee -a "$LOG"
for n in ComfyUI-GGUF ComfyUI-KJNodes ComfyUI-VideoHelperSuite; do
  R="$C/custom_nodes/$n/requirements.txt"
  if [ -f "$R" ]; then echo "-- $n" | tee -a "$LOG"; $PIP -r "$R" 2>&1 | tee -a "$LOG" || echo "WARN $n (non-fatal)" | tee -a "$LOG"; fi
done

echo "=== [5] VACE inference deps ===" | tee -a "$LOG"
$PIP ftfy pycocotools 2>&1 | tee -a "$LOG"

echo "=== [6] PINS LAST (override) ===" | tee -a "$LOG"
$PIP "transformers==4.49.0" "diffusers==0.33.1" 2>&1 | tee -a "$LOG"
$PIP --force-reinstall --no-deps "nvidia-nccl-cu12==2.28.9" 2>&1 | tee -a "$LOG"
$PIP "numpy<2" 2>&1 | tee -a "$LOG"

echo "=== [7] VERIFY ===" | tee -a "$LOG"
$PY -c "
import torch, ray, diffusers, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('ray', ray.__version__)
print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)
from torch.distributed.fsdp import fully_shard
import inspect
sig = inspect.signature(fully_shard)
print('fully_shard params:', list(sig.parameters.keys()))
assert 'ignored_params' in sig.parameters, 'FSDP2 ignored_params missing — torch too old'
print('VERIFY_OK')
" 2>&1 | tee -a "$LOG"
echo "=== INSTALL DONE $(date -u) ===" | tee -a "$LOG"
