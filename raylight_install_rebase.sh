#!/bin/bash
# Rebase-variant install: base image (runpod/pytorch torch280) ALREADY ships torch 2.8.0
# (+torchvision 0.23.0 +torchaudio 2.8.0, cu128). We install everything else and DO NOT
# touch torch — the build gate asserts 2.8.0 survived dependency resolution.
set -euo pipefail
C="${COMFY_DIR:-/opt/ComfyUI}"
PY="$(command -v python3.11 || command -v python3)"
# Constraints pin the base image's torch trio on EVERY pip call — first rebase build failed
# exactly here: an unpinned dep upgraded torch to 2.12.1+cu130 past the base's 2.8.0.
CONS=/opt/torch-constraints.txt
printf 'torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
' > "$CONS"
PIP="$PY -m pip install -c $CONS"
LOG=/root/install.log
: > "$LOG"
echo "=== REBASE INSTALL START $(date -u) python=$PY ===" | tee -a "$LOG"

echo "=== [1] Raylight deps ===" | tee -a "$LOG"
$PIP "ray>=2.48.0" "xfuser>=0.4.4" kernels huggingface_hub hf_transfer 2>&1 | tee -a "$LOG"

echo "=== [2] ComfyUI core reqs ===" | tee -a "$LOG"
$PIP -r "$C/requirements.txt" 2>&1 | tee -a "$LOG"

echo "=== [3] node reqs ===" | tee -a "$LOG"
for n in ComfyUI-GGUF ComfyUI-KJNodes ComfyUI-VideoHelperSuite RES4LYF; do
  R="$C/custom_nodes/$n/requirements.txt"
  if [ -f "$R" ]; then echo "-- $n" | tee -a "$LOG"; $PIP -r "$R" 2>&1 | tee -a "$LOG" || echo "WARN $n (non-fatal)" | tee -a "$LOG"; fi
done

echo "=== [4] VACE inference deps ===" | tee -a "$LOG"
$PIP ftfy pycocotools 2>&1 | tee -a "$LOG"

echo "=== [5] PINS (override) ===" | tee -a "$LOG"
$PIP "transformers==4.49.0" "diffusers==0.33.1" 2>&1 | tee -a "$LOG"
$PIP "numpy<2" 2>&1 | tee -a "$LOG"
$PIP runpod 2>&1 | tee -a "$LOG"

echo "=== [6] VERIFY (torch must still be the base's 2.8.0 — we never installed it) ===" | tee -a "$LOG"
$PY -c "
import torch, ray, diffusers, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda)
assert torch.__version__.startswith('2.8.0'), f'torch replaced: {torch.__version__}'
from torch.distributed.fsdp import fully_shard
import inspect
assert 'ignored_params' in inspect.signature(fully_shard).parameters
print('VERIFY_OK')
" 2>&1 | tee -a "$LOG"
echo "=== REBASE INSTALL DONE $(date -u) ===" | tee -a "$LOG"
