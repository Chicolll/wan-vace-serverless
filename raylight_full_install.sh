#!/bin/bash
# Raylight env recipe for FSDP fp8: PyTorch 2.8 + NCCL 2.28.9 into SYSTEM python3.11.
# Adapted from raylight_build_28.sh (proven on pod) into Dockerfile-friendly form.
set -u
C="${COMFY_DIR:-/opt/ComfyUI}"
PY=/usr/bin/python3.11
PIP="$PY -m pip install"
LOG=/root/install.log
: > "$LOG"
echo "=== INSTALL START $(date -u) ===" >> "$LOG"

echo "=== [1] torch 2.8 cu126 ===" >> "$LOG"
$PIP torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 >> "$LOG" 2>&1 && echo "OK_TORCH" >> "$LOG" || echo "FAIL_TORCH" >> "$LOG"

echo "=== [2] Raylight deps ===" >> "$LOG"
$PIP "ray>=2.48.0" "xfuser>=0.4.4" kernels huggingface_hub hf_transfer >> "$LOG" 2>&1 && echo "OK_DEPS" >> "$LOG" || echo "FAIL_DEPS" >> "$LOG"

echo "=== [3] ComfyUI core reqs ===" >> "$LOG"
$PIP -r "$C/requirements.txt" >> "$LOG" 2>&1 && echo "OK_COMFY" >> "$LOG" || echo "FAIL_COMFY" >> "$LOG"

echo "=== [4] node reqs ===" >> "$LOG"
for n in ComfyUI-GGUF ComfyUI-KJNodes ComfyUI-VideoHelperSuite; do
  R="$C/custom_nodes/$n/requirements.txt"
  if [ -f "$R" ]; then echo "-- $n" >> "$LOG"; $PIP -r "$R" >> "$LOG" 2>&1 || echo "WARN $n" >> "$LOG"; fi
done
echo "OK_NODES" >> "$LOG"

echo "=== [5] VACE inference deps ===" >> "$LOG"
$PIP ftfy pycocotools >> "$LOG" 2>&1 && echo "OK_VACE" >> "$LOG" || echo "FAIL_VACE" >> "$LOG"

echo "=== [6] PINS LAST (override) ===" >> "$LOG"
$PIP "transformers==4.49.0" "diffusers==0.33.1" >> "$LOG" 2>&1 && echo "OK_PINS" >> "$LOG" || echo "FAIL_PINS" >> "$LOG"
$PIP --force-reinstall --no-deps "nvidia-nccl-cu12==2.28.9" >> "$LOG" 2>&1 && echo "OK_NCCL" >> "$LOG" || echo "FAIL_NCCL" >> "$LOG"
$PIP "numpy<2" >> "$LOG" 2>&1 && echo "OK_NUMPY" >> "$LOG" || echo "FAIL_NUMPY" >> "$LOG"

echo "=== [7] VERIFY ===" >> "$LOG"
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
" >> "$LOG" 2>&1 && echo "IMPORTS_OK" >> "$LOG" || echo "IMPORTS_FAIL" >> "$LOG"
echo "=== INSTALL DONE $(date -u) ===" >> "$LOG"
