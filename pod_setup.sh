#!/bin/bash
# pod_setup.sh — reproduce the v3 Raylight FSDP env on a LIVE RunPod pod, for interactive
# multi-GPU testing WITHOUT serverless. This is the Dockerfile's setup, run by hand.
#
# Run ONCE per fresh pod. The base image MUST match the Dockerfile or the pinned install breaks:
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04   (python3.11 + CUDA 12.4)
#
# Usage:
#   git clone https://github.com/annawang7199/wan-vace-serverless.git && cd wan-vace-serverless
#   git checkout v3
#   bash pod_setup.sh
set -euo pipefail
export COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== [1] ComfyUI + custom nodes (Raylight pinned + GGUF + KJNodes + VHS + RES4LYF) =="
[ -d "$COMFY_DIR" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
cd "$COMFY_DIR/custom_nodes"
[ -d raylight ]                  || { git clone https://github.com/komikndr/raylight && ( cd raylight && git checkout ec3ac78 ); }
[ -d ComfyUI-GGUF ]             || git clone --depth 1 https://github.com/city96/ComfyUI-GGUF
[ -d ComfyUI-KJNodes ]          || git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes
[ -d ComfyUI-VideoHelperSuite ] || git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
[ -d RES4LYF ]                  || git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF   # provides res_2s

echo "== [2] Torchaudio/LTX ABI patch (Raylight __init__) =="
python3.11 "$HERE/patch_raylight_init.py"

echo "== [3] torch 2.8 / FSDP2 / ray / xfuser env — the pinned, proven recipe (the fragile part) =="
COMFY_DIR="$COMFY_DIR" bash "$HERE/raylight_full_install.sh"

echo "== [4] modified Raylight FSDP files (pre-shard support) =="
cp "$HERE/model_patcher_remote.py" "$COMFY_DIR/custom_nodes/raylight/model_patcher_remote.py"
cp "$HERE/fsdp_utils_remote.py"    "$COMFY_DIR/custom_nodes/raylight/fsdp_utils_remote.py"

cat <<DONE

POD SETUP DONE.
Launch ComfyUI (multi-GPU aware):
  cd $COMFY_DIR && python3.11 main.py --listen 0.0.0.0 --port 8188

Then open the pod's port-8188 URL, load raylight_vace_wf.json, set:
  - RayInitializer GPU = 2, ulysses_degree = 2, FSDP = true
  - WanVaceToVideo length = <your frame count, e.g. 150 for 5s @30fps>
  - the three input files (control video / *_mask_inverted.mp4 / reference png) on the volume
  - sampler stays res_2s / beta57 (RES4LYF is installed)
Queue it. 🔴 STOP the pod the moment the test finishes.
DONE
