"""Container-start model wiring for the preprocess endpoint (2026-08-19, repo split 2026-08-22).

Replaces the dockerStartCmd shell approach with an importable, testable step
the handler runs before serving. Resolves the host-NVMe model store
(Chicolll/bg-replace-preproc — default-branch HEAD = the prep set; legacy:
bg-replace-pipeline edit-preproc) if this host has it staged,
builds a local models tree of symlinks matching the graph's reference names
(incl. the `qwen/` subpaths for GGUF + LoRA), and persists the controlnet_aux
checkpoint dir on the network volume so the depth model downloads once, not
per worker.

Echoes PREP_HOSTSTORE_ACTIVE|ABSENT for log verification — a worker running
off slow volume reads must be VISIBLE, never silent.
"""
import glob
import os

VOL = "/runpod-volume"
# RunPod model-store layout: /runpod/model-store/huggingface/<org>/<repo>/<pin>/snapshots/<sha>/…
# The prep set lives in its OWN repo since 2026-08-22 (bg-replace-preproc: default-branch HEAD = the
# 5-file set — only default-branch-HEAD pins ever staged; the edit-preproc branch pin wedged 3x).
# The old repo glob stays as a fallback for hosts staged before the split.
HS_GLOBS = (
    "/runpod/model-store/huggingface/Chicolll/bg-replace-preproc/*/snapshots/*/",
    "/runpod/model-store/huggingface/Chicolll/bg-replace-pipeline/*/snapshots/*/",
)
HS_GLOB = HS_GLOBS[0]  # kept for callers that import the old name
LOCAL = "/opt/prep_models"          # what extra_model_paths.yaml points at
VOL_MODELS = f"{VOL}/prep-models/ComfyUI/models"  # volume fallback tree
AUX_CKPTS_VOL = f"{VOL}/prep-models/aux-ckpts"    # controlnet_aux ckpts dir (writable; depth ckpt lands here on a miss)
AUX_HS_SUB = "aux"   # <snapshot>/aux/<hf-repo-id>/<file> mirrors controlnet_aux's ckpts/<repo_id>/<file> layout
COMFY_DIR = os.environ.get("COMFY_DIR", "/opt/ComfyUI")


def _link(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)


def _wire_aux(hs):
    """controlnet_aux resolves ckpts/<hf-repo-id>/<file> and downloads from Hugging Face on a miss
    (custom_hf_download). ckpts stays the volume dir (writable, survives workers). When the host
    store snapshot carries an aux/ tree, every file in it is exposed inside ckpts as a symlink to the
    NVMe copy — replacing any volume copy, which is a redundant download cache. When it does not,
    dangling links left by a host that had one are removed so the pack can re-download.
    Returns (wired, cleaned) path lists."""
    wired, cleaned = [], []
    aux_src = os.path.join(hs, AUX_HS_SUB) if hs else None
    if aux_src and os.path.isdir(aux_src):
        for dp, _, fs in os.walk(aux_src):
            for fn in fs:
                src = os.path.join(dp, fn)
                dst = os.path.join(AUX_CKPTS_VOL, os.path.relpath(src, aux_src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.islink(dst):
                    if os.readlink(dst) == src:
                        wired.append(dst)
                        continue
                    os.unlink(dst)
                elif os.path.exists(dst):
                    print(f"prep_setup: removing volume copy {dst} (superseded by host store)", flush=True)
                    os.remove(dst)
                os.symlink(src, dst)
                wired.append(dst)
    if os.path.isdir(AUX_CKPTS_VOL):
        for dp, _, fs in os.walk(AUX_CKPTS_VOL):
            for fn in fs:
                q = os.path.join(dp, fn)
                if os.path.islink(q) and not os.path.exists(q):
                    os.unlink(q)
                    cleaned.append(q)
    return wired, cleaned


def run() -> str:
    # Hosts are shared with the render endpoint, which stages a DIFFERENT
    # snapshot of the same HF repo (video-only, no sam3/qwen). Only snapshots
    # that actually contain the prep set qualify; freshest of those wins.
    snaps = sorted(
        (s for g in HS_GLOBS for s in glob.glob(g) if os.path.isdir(os.path.join(s, "sam3"))),
        key=os.path.getmtime,
    )
    hs = snaps[-1] if snaps else None

    for sub in ("unet", "loras", "text_encoders", "vae", "sam3"):
        os.makedirs(os.path.join(VOL_MODELS, sub), exist_ok=True)
    os.makedirs(AUX_CKPTS_VOL, exist_ok=True)

    # controlnet_aux downloads its checkpoints into <pack>/ckpts by default —
    # point that at the volume so depth_anything_v2_vitl.pth survives workers.
    aux_ckpts = os.path.join(COMFY_DIR, "custom_nodes", "comfyui_controlnet_aux", "ckpts")
    if not os.path.islink(aux_ckpts):
        if os.path.isdir(aux_ckpts) and not os.listdir(aux_ckpts):
            os.rmdir(aux_ckpts)
        if not os.path.exists(aux_ckpts):
            os.symlink(AUX_CKPTS_VOL, aux_ckpts)

    src = hs if hs else VOL_MODELS
    # Graph references: unet+gguf see 'qwen/<gguf>', loras sees 'qwen/<lora>',
    # text_encoders/clip and vae and sam3 are flat.
    _link(os.path.join(src, "unet"), os.path.join(LOCAL, "unet", "qwen"))
    _link(os.path.join(src, "loras"), os.path.join(LOCAL, "loras", "qwen"))
    _link(os.path.join(src, "text_encoders"), os.path.join(LOCAL, "text_encoders"))
    _link(os.path.join(src, "vae"), os.path.join(LOCAL, "vae"))
    _link(os.path.join(src, "sam3"), os.path.join(LOCAL, "sam3"))
    # Aux checkpoints: host-store copies linked into ckpts; the ckpts dir itself is linked under
    # LOCAL so the boot pre-warm walk covers any file that is still volume-resident.
    wired, cleaned = _wire_aux(hs)
    _link(AUX_CKPTS_VOL, os.path.join(LOCAL, AUX_HS_SUB))

    aux_state = f"aux={len(wired)}/hoststore" if wired else "aux=volume"
    if cleaned:
        aux_state += f" (cleaned {len(cleaned)} dangling)"
    state = f"PREP_HOSTSTORE_{'ACTIVE ' + hs if hs else 'ABSENT (volume fallback ' + VOL_MODELS + ')'} {aux_state}"
    print(state, flush=True)
    return state


if __name__ == "__main__":
    run()
