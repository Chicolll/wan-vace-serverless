# v3 — Anna's no-chunk fork (branch `v3`, forked from `fsdp`)

Goal: adopt the partner's seam-free multi-GPU render (SkyReels fp8 + FSDP + USP, single-pass,
no chunking) while keeping **our** quality sampler and front-end.

## What changed vs `fsdp`
1. **Sampler → `res_2s` / `beta57` / 8 steps** (`raylight_vace_wf.json` node 15), replacing the
   partner's `euler` / `beta` / 6. This is our validated quality sampler for hard scenes
   (underwater, caustics, fast motion) where euler/6 was only proven on easy footage.
2. **RES4LYF installed in the image** (`Dockerfile` + `raylight_full_install.sh`) — this is the
   node pack that *provides* `res_2s`. Without it, `res_2s` doesn't exist.
3. **Per-job sampler override** (`handler_raylight.py` `_build_wf`): pass `"sampler_name"` /
   `"scheduler"` in the job to A/B euler vs res_2s **without rebuilding the image**. Defaults to
   the JSON's res_2s/beta57. Default `sample_steps` bumped 6 → 8 to match res_2s.

## ⚠️ Open question to validate after deploy
Does `XFuserKSamplerAdvanced` (the USP sampler) actually accept `res_2s`? RES4LYF registers it
into ComfyUI's sampler list, but the xDiT USP path may only support its own sampler set. **Test
on first render.** If it rejects res_2s, fall back per-job: `"sampler_name": "euler", "sample_steps": 6`
(no rebuild needed), and we accept euler on the multi-GPU path.

## Preserved from the partner (unchanged)
- Model: `wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors` (fp8) — **same SkyReels merge we already run** (Q4) locally.
- FSDP 2-GPU + USP, single-pass, **no chunking**.
- VACE inputs: `control_video` (node 9), `control_masks` (node 10), `reference_image` (node 12).
  Our **3 control modes (depth / canny / camera-track), SAM3 masking, and Qwen start-frame stay in
  PREPROCESSING** — they produce the control video / `*_mask_inverted.mp4` / reference PNG that this
  render consumes. The render itself is control-agnostic.

## Remaining to go live (deploy steps, need resources)
1. Push this branch to our GitHub fork (needs our GitHub auth — `gh`/token).
2. Build the Docker image (~15-20 min) + push to a registry.
3. Stage the fp8 SkyReels model on our RunPod volume (17.3 GB, HF pod-fetch).
4. Create a 2-GPU FSDP endpoint (idle-timeout 5s = no idle billing).
5. Build the preprocessing step (SAM3 + depth/canny/camera-track + Qwen → the 3 input files).
6. Cut an 81-frame clip → preprocess → debug payload → render → compare vs our chunked baseline.
