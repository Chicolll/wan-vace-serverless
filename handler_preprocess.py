"""RunPod serverless handler — PREPROCESS endpoint ("prep kitchen", 2026-07-15).

Deliberately dumb: receives a fully-built ComfyUI graph (built client-side by
aivfx-pipeline/preprocess_graph.py), runs it, renames the outputs to exact final
names in INPUTS_DIR (the flat folder the render endpoint reads), returns the file
list. No graph logic lives here — recipe changes never require an image rebuild.

Job input:
  {"graph": {...},                          # ComfyUI API-format prompt graph
   "outputs": {"<needle>": "<final.mp4>"},  # substring -> exact final filename
   "timeout_s": 1500}                       # optional
  or {"debug": true}                        # env/dir sanity check, no GPU work
  or {"fetch": {"url": "...", "dest": "ComfyUI/models/sam3/sam3.pt"}}
                                            # download big files straight to the volume
                                            # (local S3 uploads cap at ~100MB — stage
                                            # large model weights through this op)

Output files land in INPUTS_DIR (default /runpod-volume/native-xdit/inputs) —
flat folder, collision-proof names are the CLIENT's job (bake the date into the
tag, e.g. DRIVING-D2-0715-BAKED.mp4). See PREPROCESS_ENDPOINT_SPEC.md.
"""
import json, os, shutil, subprocess, sys, time
import urllib.request

import runpod

VOL        = "/runpod-volume"
COMFY_DIR  = os.environ.get("COMFY_DIR", "/opt/ComfyUI")
INPUTS_DIR = os.environ.get("INPUTS_DIR", f"{VOL}/native-xdit/inputs")
STAGE_DIR  = os.environ.get("STAGE_DIR", f"{VOL}/native-xdit/prep_stage")
EMP        = os.environ.get("EXTRA_MODEL_PATHS", "/opt/extra_model_paths.yaml")
PORT       = int(os.environ.get("COMFY_PORT", "8188"))
URL        = f"http://127.0.0.1:{PORT}"
LOG        = "/tmp/comfy_prep.log"

_comfy = None


def _tail(path, n=40):
    try:
        with open(path, errors="replace") as f:
            return "\n".join(f.read().splitlines()[-n:])
    except OSError:
        return "<no log>"


def _http(path, payload=None, timeout=30):
    req = urllib.request.Request(URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def ensure_comfy(deadline_s=240):
    """Boot ComfyUI once per worker; reuse the warm process across jobs."""
    global _comfy
    if _comfy is not None and _comfy.poll() is None:
        return
    os.makedirs(INPUTS_DIR, exist_ok=True)
    os.makedirs(STAGE_DIR, exist_ok=True)
    env = dict(os.environ)
    env["COMFY_DIR"] = COMFY_DIR
    env["COMFY_ARGS"] = (f"--listen 127.0.0.1 --port {PORT} --disable-auto-launch "
                         f"--input-directory {INPUTS_DIR} --output-directory {STAGE_DIR} "
                         f"--extra-model-paths-config {EMP}")
    _comfy = subprocess.Popen([sys.executable, "/opt/comfy_launch.py"],
                              stdout=open(LOG, "w"), stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if _comfy.poll() is not None:
            raise RuntimeError(f"ComfyUI exited rc={_comfy.returncode}. log tail:\n{_tail(LOG)}")
        try:
            _http("/system_stats"); return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"ComfyUI boot timeout. log tail:\n{_tail(LOG)}")


def _find_newest(root, needle):
    best, best_m = None, -1
    for dirpath, _, files in os.walk(root):
        for f in files:
            if needle in f:
                p = os.path.join(dirpath, f)
                m = os.path.getmtime(p)
                if m > best_m: best, best_m = p, m
    return best


def handler(job):
    j = job.get("input") or {}
    if j.get("debug"):
        return {"comfy_dir": COMFY_DIR, "inputs_dir": INPUTS_DIR,
                "inputs_dir_exists": os.path.isdir(INPUTS_DIR),
                "stage_dir": STAGE_DIR, "emp_exists": os.path.isfile(EMP),
                "inputs_sample": sorted(os.listdir(INPUTS_DIR))[:20] if os.path.isdir(INPUTS_DIR) else []}
    if j.get("fetch"):
        url, dest = j["fetch"]["url"], j["fetch"]["dest"]
        if ".." in dest or dest.startswith("/"):
            return {"error": "dest must be a relative volume path"}
        path = os.path.join(VOL, dest)
        if os.path.exists(path) and not j["fetch"].get("overwrite"):
            return {"fetched": dest, "bytes": os.path.getsize(path), "skipped": "already exists"}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.time()
        tmp = path + ".part"
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        os.replace(tmp, path)
        return {"fetched": dest, "bytes": os.path.getsize(path), "secs": round(time.time() - t0, 1)}
    graph, outputs = j.get("graph"), j.get("outputs") or {}
    if not graph or not outputs:
        return {"error": "need input.graph and input.outputs"}
    t0 = time.time()
    ensure_comfy()
    t_boot = time.time()

    # fresh stage subdir per job so needle matching can't hit stale files
    stage = os.path.join(STAGE_DIR, f"job_{int(t0)}")
    for nid, node in graph.items():
        pref = node.get("inputs", {}).get("filename_prefix")
        if pref is not None:
            node["inputs"]["filename_prefix"] = f"job_{int(t0)}/" + pref

    pid = _http("/prompt", {"prompt": graph}).get("prompt_id")
    if not pid:
        return {"error": "submit failed", "log_tail": _tail(LOG)}
    deadline = time.time() + int(j.get("timeout_s", 1500))
    while True:
        h = _http(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                break
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                return {"error": "graph execution error",
                        "detail": json.dumps(msgs)[:1500], "log_tail": _tail(LOG)}
        if time.time() > deadline:
            _http("/interrupt", {})
            return {"error": "graph timeout", "log_tail": _tail(LOG)}
        time.sleep(5)
    t_graph = time.time()

    written, missing = [], []
    for needle, final in outputs.items():
        src = _find_newest(stage, needle) or _find_newest(STAGE_DIR, needle)
        if not src:
            missing.append(needle); continue
        dst = os.path.join(INPUTS_DIR, final)
        shutil.move(src, dst)
        written.append({"file": final, "bytes": os.path.getsize(dst)})
    shutil.rmtree(stage, ignore_errors=True)
    if missing:
        return {"error": "missing outputs", "missing": missing,
                "written": written, "log_tail": _tail(LOG)}
    return {"written": written,
            "timing": {"boot_s": round(t_boot - t0, 1),
                       "graph_s": round(t_graph - t_boot, 1),
                       "total_s": round(time.time() - t0, 1)}}


runpod.serverless.start({"handler": handler})
