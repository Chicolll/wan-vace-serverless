import os, sys, runpy, shlex
# Path-flexible launcher. Pod default kept; serverless overrides via COMFY_DIR / COMFY_ARGS env.
COMFY_DIR = os.environ.get("COMFY_DIR", "/workspace/runpod-slim/ComfyUI")
# ComfyUI parses sys.argv at `import comfy.cli_args`, so set argv BEFORE importing it.
sys.argv = [os.path.join(COMFY_DIR, "main.py")] + shlex.split(os.environ.get("COMFY_ARGS", ""))
sys.path.insert(0, COMFY_DIR)
import comfy.cli_args
# Raylight rev ec3ac78 reads comfy_args.vram_headroom which the pinned ComfyUI lacks.
# Inject a safe default on the main-process args namespace (workers guard with hasattr already).
if not hasattr(comfy.cli_args.args, "vram_headroom"):
    setattr(comfy.cli_args.args, "vram_headroom", None)
runpy.run_path(sys.argv[0], run_name="__main__")
