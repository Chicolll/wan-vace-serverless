#!/usr/bin/env python3
"""Patch Raylight __init__.py to wrap torchaudio-dependent imports in try/except.

nodes_lt and comfyui_ltxv load torchaudio operators that ABI-mismatch when the base
Docker image has a different torch version than what we install (e.g., base=2.4, we
install=2.8). These are LTX audio/video nodes, not needed for VACE workflows.

Usage (in Dockerfile):  RUN python3.11 /opt/patch_raylight_init.py
"""
import re, sys

INIT = "/opt/ComfyUI/custom_nodes/raylight/__init__.py"
MODULES = ["nodes_lt", "comfyui_ltxv"]

STUB = 'type("M", (), {"NODE_CLASS_MAPPINGS": {}, "NODE_DISPLAY_NAME_MAPPINGS": {}})()'

src = open(INIT).read()
patched = 0

for mod in MODULES:
    pattern = rf'^(\s*)(from \. import {mod}\b.*)$'
    match = re.search(pattern, src, re.MULTILINE)
    if not match:
        print(f"WARN: 'from . import {mod}' not found in {INIT} — skipping")
        continue
    indent = match.group(1)
    orig_line = match.group(2)
    replacement = (
        f"{indent}try:\n"
        f"{indent}    {orig_line}\n"
        f"{indent}except Exception:\n"
        f"{indent}    {mod} = {STUB}"
    )
    src = src[:match.start()] + replacement + src[match.end():]
    patched += 1
    print(f"Patched: {mod}")

open(INIT, "w").write(src)
print(f"Done: {patched}/{len(MODULES)} imports wrapped in {INIT}")
if patched == 0:
    print("Nothing to patch — modules not present in this Raylight version (safe to skip)")
