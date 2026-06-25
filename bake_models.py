#!/usr/bin/env python3
"""Bake the 4 workflow models into the image at /opt/ComfyUI/models (LOCAL container disk).

Why baked, not volume-mounted: the cold load is ~80% network-volume (MooseFS) READ (~115s on H100). Putting the
models on the image's local disk removes that read entirely — every cold worker reads from local disk, and RunPod
caches image layers per host (FlashBoot). It also sidesteps RunPod's one-cached-model-per-endpoint limit (we need 4
files from 3 repos). All public HF repos, sha256-verified (the same bytes the manifest staged to the volume); a
mismatch FAILS the BUILD (never a live, metered worker). ~19GB total.
"""
import hashlib
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

COMFY = os.environ.get("COMFY_DIR", "/opt/ComfyUI")
TMP = "/tmp/bake_dl"

# (repo_id, repo filename, ComfyUI models/ subdir, expected sha256) — repos+hashes from the proven download manifest.
MODELS = [
    ("mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF",
     "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf", "unet/gguf",
     "3e6818d87c6659be95b10281b56bb46abd3fe41d7d28f2d124401a64d00f4609"),
    ("DeepBeepMeep/Wan2.1",
     "loras_accelerators/Wan2.1_T2V_14B_FusionX_LoRA.safetensors", "loras",
     "c653087fa7c9163abdd1a4c627bfc483966c6d78a4426667d1185f860951c30e"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders",
     "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/vae/wan_2.1_vae.safetensors", "vae",
     "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for repo, fn, sub, want in MODELS:
        dst_dir = os.path.join(COMFY, "models", sub)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(fn))
        print(f"BAKE {repo}::{fn} -> {dst}", flush=True)
        src = hf_hub_download(repo_id=repo, filename=fn, local_dir=TMP)
        shutil.move(src, dst)
        got = sha256(dst)
        if got != want:
            print(f"SHA256_MISMATCH {dst}: {got} != {want}", file=sys.stderr)
            sys.exit(1)
        print(f"BAKED_OK {dst} ({os.path.getsize(dst)} bytes)", flush=True)
        shutil.rmtree(TMP, ignore_errors=True)
    print("ALL_MODELS_BAKED", flush=True)


if __name__ == "__main__":
    main()
