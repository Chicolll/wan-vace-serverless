"""Build-time node-class gate (2026-08-19).

Boots ComfyUI on CPU inside `docker build`, reads /object_info, and asserts
every node class the preprocess graph uses is registered. A broken node pack
or a ComfyUI pin that dropped a class fails the BUILD — the cheapest possible
place to fail. Also bakes the SQLite migrations (the CPU boot creates the DB,
saving ~1.5 s per container start, measured on the render side 7/15).
"""
import json
import subprocess
import sys
import time
import urllib.request

REQUIRED = [
    "CFGNorm", "CLIPLoader", "CannyEdgePreprocessor", "DepthAnythingV2Preprocessor",
    "EmptyImage", "FluxKontextImageScale", "ImageBlend", "ImageCompositeMasked",
    "ImageFromBatch", "ImageResizeKJv2", "ImageToMask", "InvertMask", "KSampler",
    "LoaderGGUF", "LoraLoaderModelOnly", "MaskAreaFilter", "MaskToImage",
    "ModelSamplingAuraFlow", "RepeatImageBatch", "SAM3Segment", "SaveImage",
    "TextEncodeQwenImageEditPlus", "VAEDecode", "VAEEncode", "VAELoader",
    "VHS_LoadVideo", "VHS_VideoCombine",
]

proc = subprocess.Popen(
    [sys.executable, "/opt/ComfyUI/main.py", "--cpu", "--listen", "127.0.0.1",
     "--port", "8188", "--disable-auto-launch"],
    cwd="/opt/ComfyUI", stdout=open("/tmp/gate_boot.log", "w"), stderr=subprocess.STDOUT)

deadline = time.time() + 300
info = None
while time.time() < deadline:
    if proc.poll() is not None:
        print(open("/tmp/gate_boot.log", errors="replace").read()[-4000:])
        sys.exit(f"ComfyUI exited rc={proc.returncode} during gate")
    try:
        info = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=10))
        break
    except Exception:
        time.sleep(3)
if info is None:
    print(open("/tmp/gate_boot.log", errors="replace").read()[-4000:])
    sys.exit("ComfyUI never answered /object_info within 300s")

missing = [c for c in REQUIRED if c not in info]
proc.terminate()
if missing:
    print(open("/tmp/gate_boot.log", errors="replace").read()[-4000:])
    sys.exit(f"NODE GATE FAILED — missing classes: {missing}")
print(f"NODE_GATE_OK {len(REQUIRED)} classes present")
