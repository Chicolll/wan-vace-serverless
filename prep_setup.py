"""Container-start model wiring for the preprocess endpoint (2026-08-19).

Replaces the dockerStartCmd shell approach with an importable, testable step
the handler runs before serving. Resolves the host-NVMe model store
(Chicolll/bg-replace-pipeline, edit-preproc pin) if this host has it staged,
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
HS_GLOB = "/runpod/model-store/huggingface/Chicolll/bg-replace-pipeline/*/snapshots/*/"
LOCAL = "/opt/prep_models"          # what extra_model_paths.yaml points at
VOL_MODELS = f"{VOL}/prep-models/ComfyUI/models"  # volume fallback tree
AUX_CKPTS_VOL = f"{VOL}/prep-models/aux-ckpts"    # depth ckpt persistence
COMFY_DIR = os.environ.get("COMFY_DIR", "/opt/ComfyUI")


def _link(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)


def run() -> str:
    # Hosts are shared with the render endpoint, which stages a DIFFERENT
    # snapshot of the same HF repo (video-only, no sam3/qwen). Only snapshots
    # that actually contain the prep set qualify; freshest of those wins.
    snaps = sorted(
        (s for s in glob.glob(HS_GLOB) if os.path.isdir(os.path.join(s, "sam3"))),
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

    state = f"PREP_HOSTSTORE_{'ACTIVE ' + hs if hs else 'ABSENT (volume fallback ' + VOL_MODELS + ')'}"
    print(state, flush=True)
    return state


if __name__ == "__main__":
    run()
