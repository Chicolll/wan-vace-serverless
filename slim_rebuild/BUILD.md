# True-slim image build — how to run it

Self-contained build context (this directory). Everything pinned from the LIVE container's
7/14 capture (`notes/slim_image_runtime_matrix.md` + `requirements.lock`); the old cu126
Dockerfile is NOT a source of truth.

## Build route (pick one)
1. **GitHub Actions (preferred):** push this directory to a NEW branch (never `fsdp`/`raylight`)
   of the code repo with a workflow that runs `docker build` + push to
   `ghcr.io/chicolll/wan-vace-fsdp:trueslim-<shortsha>` using the automatic GITHUB_TOKEN.
   Verify first that the RunPod GitHub integration only watches its connected branch.
2. **CPU pod + kaniko:** unprivileged build on a RunPod CPU pod (attempt CPU first per Gate 0),
   `gcr.io/kaniko-project/executor` with a GHCR push secret the user provides by hand
   (credential-relocation is classifier-blocked for the agent).
3. Local Docker Desktop if available.

## Gates (in order, all BEFORE any prod change)
1. Build passes the embedded IMPORT_GATE (torch 2.8.0+cu128, FSDP2 ignored_params present).
2. `docker buildx imagetools inspect ghcr.io/...:trueslim-<sha>` — **<=8 GiB compressed**, no
   multi-GiB duplicate CUDA/PyTorch layers. Kill here if fat.
3. One canary boot on stagingtest: clone template vp9reh1lta -> new template with the trueslim
   image (same dockerArgs), point stagingtest at it, one boot to warm + warmup ok + code_fp
   beacon matches. Compare boot phases vs the 27.2 s baseline.
4. Fresh-host pull time: first boot on a never-seen host measures the pull (expect ~60-100 s
   saved vs the 160 s / 14.86 GiB pull).
5. Prod repoint = separate explicit go (template image swap; undo = point back to slim-baked).

## Known risks
- Node packs other than raylight are cloned at HEAD (live versions weren't pinned anywhere);
  gate 3's workflow validation + code_fp beacon is the check.
- python:3.11-slim-bookworm (Debian) vs live Ubuntu base: torch manylinux wheels are ABI-safe,
  but anything importing Ubuntu system python packages would break — the lock excludes them
  (python-apt, PyGObject, launchpad stack) after verifying the handler/Comfy never import them.
- imageio-ffmpeg provides the ffmpeg binary via pip (no apt ffmpeg installed, same as live).
