#!/usr/bin/env python3
r"""
Streamlit UI for Photo Organizer (Preview/Sampler + Full Run)

Run:  streamlit run app_streamlit.py
"""

import os
import sys
import json
import time
import random
import threading
import atexit
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import socket
import asyncio

import streamlit as st
from PIL import Image, ExifTags
import torch
import open_clip

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "websockets is required. Please add `websockets>=12.0` to requirements.txt"
    ) from e

# ----------------------------
# Settings & constants
# ----------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
CHILD_PROCS: List[subprocess.Popen] = []

# WS tracking
_WS_PORT: int = 0
_WS_SERVER = None
_WS_LOOP: asyncio.AbstractEventLoop | None = None
_WS_CLIENTS: set[WebSocketServerProtocol] = set()
_WS_LOCK = threading.Lock()
_SEEN_A_CLIENT = False
_AUTO_SHUTDOWN_ENABLED = True
_GRACE_SECONDS = 6.0

# ----------------------------
# Graceful shutdown helpers
# ----------------------------

def _terminate_children():
    for p in list(CHILD_PROCS):
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except Exception:
                    p.kill()
        except Exception:
            pass
    CHILD_PROCS.clear()


aTexit_registered = False
if not aTexit_registered:
    atexit.register(_terminate_children)
    aTexit_registered = True


def request_shutdown():
    """Stop Streamlit server and exit the process (works in EXE)."""
    _terminate_children()
    # Attempt to stop Streamlit cleanly (works across many versions)
    try:
        from streamlit.web.server.server import Server  # type: ignore
        server = Server.get_current()
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass
    except Exception:
        pass
    os._exit(0)

# ----------------------------
# WebSocket infra
# ----------------------------

async def _ws_handle_client(ws: WebSocketServerProtocol):
    global _SEEN_A_CLIENT
    # Track connect
    with _WS_LOCK:
        _WS_CLIENTS.add(ws)
    _SEEN_A_CLIENT = True
    # Echo loop (keeps the connection alive)
    try:
        async for _ in ws:
            await ws.send("pong")
    except websockets.exceptions.ConnectionClosed:  # normal disconnect
        pass
    finally:
        with _WS_LOCK:
            _WS_CLIENTS.discard(ws)


async def _ws_main(port: int):
    async with websockets.serve(
        _ws_handle_client,
        host="127.0.0.1",
        port=port,
        ping_interval=2,
        ping_timeout=4,
    ) as server:
        await asyncio.Future()  # run forever


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@st.cache_resource(show_spinner=False)
def _start_ws_server() -> int:
    """Start the WS server once per process; return chosen port."""
    global _WS_LOOP
    port = _find_free_port()
    loop = asyncio.new_event_loop()
    _WS_LOOP = loop

    def _runner():
        try:
            loop.run_until_complete(_ws_main(port))
        except Exception:
            pass

    t = threading.Thread(target=_runner, daemon=True, name="ws-server")
    t.start()
    return port


def _start_ws_watcher():
    """Background watcher that exits when all clients are gone for GRACE_SECONDS."""
    def _watch():
        last_nonzero = 0.0
        while True:
            time.sleep(1.0)
            try:
                with _WS_LOCK:
                    n = len(_WS_CLIENTS)
                now = time.time()
                if n > 0:
                    last_nonzero = now
                elif _AUTO_SHUTDOWN_ENABLED and _SEEN_A_CLIENT and last_nonzero and (now - last_nonzero) >= _GRACE_SECONDS:
                    request_shutdown()
                    return
            except Exception:
                # Never kill the watcher
                pass

    threading.Thread(target=_watch, daemon=True, name="ws-watcher").start()


def _inject_ws_client(port: int):
    st.markdown(
        f"""
        <script>
        (function() {{
          if (window.__photo_ws_ready__) return; // guard per tab
          window.__photo_ws_ready__ = true;
          const WS_URL = 'ws://127.0.0.1:{port}';
          let ws = null;
          let pingTimer = null;

          function connect() {{
            try {{
              ws = new WebSocket(WS_URL);
              ws.onopen = function() {{
                try {{ if (pingTimer) clearInterval(pingTimer); }} catch (_err) {{}}
                pingTimer = setInterval(function() {{
                  if (ws && ws.readyState === WebSocket.OPEN) {{
                    try {{ ws.send('ping'); }} catch (_e) {{}}
                  }}
                }}, 2000);
              }};
              ws.onclose = function() {{
                try {{ if (pingTimer) clearInterval(pingTimer); }} catch (_err) {{}}
                // If the tab is still open, try to reconnect after a short delay
                if (!document.hidden) {{
                  setTimeout(connect, 1000);
                }}
              }};
              ws.onerror = function(_e) {{ try {{ ws.close(); }} catch (_err) {{}} }};
              window.addEventListener('pagehide', function() {{ try {{ ws.close(); }} catch (_e) {{}} }}, {{ passive: true }});
              window.addEventListener('beforeunload', function() {{ try {{ ws.close(); }} catch (_e) {{}} }}, {{ passive: true }});
            }} catch (_e) {{ /* swallow */ }}
          }}
          connect();
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Shared logic (mirrors CLI)
# ----------------------------

def load_labels(labels_path: Path) -> Dict[str, Dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for label, cfg in data.items():
        cfg.setdefault("synonyms", [])
        base_prompt = cfg.get("prompt", label)
        if not cfg["synonyms"] or base_prompt not in cfg["synonyms"]:
            cfg["synonyms"].insert(0, base_prompt)
        cfg.setdefault("weight", 1.0)
    return data


def _prompt_variants(syn: str, rich: bool) -> List[str]:
    if not rich:
        return [f"a photo of {syn}"]
    return [
        f"{syn}",
        f"a photo of {syn}",
        f"an image of {syn}",
        f"a close-up photo of {syn}",
        f"a snapshot of {syn}",
    ]


def build_text_tokens(model, tokenizer, labels_cfg: Dict[str, Dict], device: str, rich_prompts: bool, ignore_weights: bool):
    texts: List[str] = []
    owners: List[Tuple[str, float, str]] = []
    for label, cfg in labels_cfg.items():
        weight = 1.0 if ignore_weights else float(cfg.get("weight", 1.0))
        for syn in cfg.get("synonyms", []):
            for prompt_t in _prompt_variants(syn, rich_prompts):
                texts.append(prompt_t)
                owners.append((label, weight, syn))
    if not texts:
        raise RuntimeError("No text prompts from labels.json")
    text_tokens = tokenizer(texts)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens.to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, owners, texts


def load_image(path: Path, preprocess, device: str):
    from PIL import Image as _Image
    img = _Image.open(path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    return img_t


def score_image(model, image_t, text_features, owners, agg: str = "max", temperature: float = 1.0):
    with torch.no_grad():
        image_features = model.encode_image(image_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) / temperature
        probs = logits.softmax(dim=-1).squeeze(0)
    per_label: Dict[str, List[float]] = {}
    for i, p in enumerate(probs):
        lab, w, _ = owners[i]
        per_label.setdefault(lab, []).append(float(p) * float(w))
    scores: Dict[str, float] = {}
    for lab, vals in per_label.items():
        if agg == "sum":
            s = sum(vals)
        elif agg == "mean":
            s = sum(vals) / max(1, len(vals))
        else:  # max
            s = max(vals)
        scores[lab] = s
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def decide_label(scores: Dict[str, float], policy: str, threshold: float, margin: float, ratio: float, topk: int) -> List[str]:
    if not scores:
        return ["_unsorted"]
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = items[0]
    second = items[1][1] if len(items) > 1 else 0.0
    accept = False
    if policy == "threshold":
        accept = top_score >= threshold
    elif policy == "margin":
        accept = (top_score - second) >= margin
    elif policy == "ratio":
        denom = max(second, 1e-9)
        accept = (top_score / denom) >= ratio
    elif policy == "always-top1":
        accept = True
    chosen: List[str] = []
    if accept:
        chosen = [lab for lab, _ in items[: max(1, topk)]]
    else:
        chosen = [top_label] if top_score >= (threshold * 0.6) else ["_unsorted"]
    return chosen


def get_exif_year_month(path: Path) -> Tuple[int, int]:
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        date_tag = None
        for k, v in ExifTags.TAGS.items():
            if v == "DateTimeOriginal":
                date_tag = k
                break
        if date_tag and date_tag in exif:
            dt_str = exif[date_tag]
            parts = dt_str.replace(":", "-", 2)
            from datetime import datetime
            dt = datetime.fromisoformat(parts)
            return dt.year, dt.month
    except Exception:
        pass
    from datetime import datetime
    dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt.year, dt.month


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Photo Organizer — Preview & Run", layout="wide")
st.title("📷 Photo Organizer — Preview & Full Run")

# Start WebSocket server/watcher once per process
_WS_PORT = _start_ws_server()
_start_ws_watcher()
_inject_ws_client(_WS_PORT)

with st.sidebar:
    st.header("Settings")
    src = Path(st.text_input("Source folder (--src)", value=str(Path.cwd())))
    dst = Path(st.text_input("Destination (--dst)", value=str(Path.cwd() / "sorted")))
    labels_path = Path(st.text_input("Labels file (--labels)", value=str(Path("labels.json").resolve())))

    st.subheader("Routing")
    agg = st.selectbox("Aggregation (--agg)", ["max", "mean", "sum"], index=0)
    decision = st.selectbox("Decision (--decision)", ["margin", "ratio", "threshold", "always-top1"], index=0)
    margin = st.slider("Margin (--margin)", 0.0, 0.03, 0.008, 0.001)
    ratio = st.slider("Ratio (--ratio)", 1.00, 1.30, 1.06, 0.01)
    threshold = st.slider("Threshold (--threshold)", 0.05, 0.4, 0.10, 0.01)

    st.subheader("Behavior")
    topk = st.selectbox("Top-K (--topk)", [1, 2, 3], index=0)
    copy = st.checkbox("Copy (not move) (--copy)", value=False)
    dedupe = st.checkbox("Deduplicate (--dedupe)", value=False)
    rich_prompts = st.checkbox("Rich prompts (--rich-prompts)", value=True)
    ignore_weights = st.checkbox("Ignore weights (--ignore-weights)", value=True)

    # New flags
    no_overwrite = st.checkbox("No-overwrite on copy (--no-overwrite)", value=False,
                               help="When copying, never overwrite; create unique names like file__1.jpg")
    skip_existing = st.checkbox("Skip existing (--skip-existing)", value=False,
                                help="If the target path exists, skip instead of writing")
    scan_dst = st.checkbox("Scan dst for dupes (--scan-dst-dedupe)", value=False,
                           help="Pre-scan destination to skip images that already exist there (by perceptual hash)")

    st.subheader("Lifecycle")
    auto_shutdown = st.checkbox(
        "Auto-stop app when all tabs close",
        value=True,
        help="Exit a few seconds after the last tab disconnects.",
    )
    if st.button("⏹️ Shutdown app now"):
        st.warning("Shutting down…")
        request_shutdown()

    # Debug block
    with st.expander("Session/WS debug"):
        with _WS_LOCK:
            n = len(_WS_CLIENTS)
        st.write({
            "ws_port": _WS_PORT,
            "connected_tabs": n,
            "auto_shutdown": auto_shutdown,
            "seen_a_client": _SEEN_A_CLIENT,
            "grace_seconds": _GRACE_SECONDS,
        })

    st.subheader("Preview sample")
    sample_n = st.number_input("Sample size", min_value=1, max_value=500, value=50, step=1)
    do_preview = st.button("🔎 Preview sample (dry)")

    st.subheader("Full run")
    run_copy = st.button("▶ Run FULL (COPY)")
    run_move = st.button("⚠ Run FULL (MOVE)")

# Update toggle
_AUTO_SHUTDOWN_ENABLED = bool(auto_shutdown)

# Safeguards
if not src.exists():
    st.warning("Source folder does not exist.")
if not labels_path.exists():
    st.warning("labels.json not found.")

# Model init (lazy)
@st.cache_resource(show_spinner=True)
def get_model_and_text_features(labels_path: Path, rich_prompts: bool, ignore_weights: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    pretrained = "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    labels_cfg = load_labels(labels_path)
    text_features, owners, prompts = build_text_tokens(
        model, tokenizer, labels_cfg, device, rich_prompts, ignore_weights
    )
    return {
        "device": device,
        "model": model,
        "preprocess": preprocess,
        "text_features": text_features,
        "owners": owners,
        "prompts": prompts,
    }


# Collect images
@st.cache_data(show_spinner=False)
def list_images(folder: Path) -> List[Path]:
    imgs: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            imgs.append(p)
    return imgs


# PREVIEW MODE ---------------------------------------------------------------
if do_preview and src.exists() and labels_path.exists():
    imgs = list_images(src)
    if not imgs:
        st.info("No supported images found in source folder.")
    else:
        import random as _rnd
        sample = imgs if len(imgs) <= sample_n else _rnd.sample(imgs, sample_n)
        st.write(f"Previewing **{len(sample)}** of **{len(imgs)}** images from `{src}`.")

        bundle = get_model_and_text_features(labels_path, rich_prompts, ignore_weights)
        model = bundle["model"]
        preprocess = bundle["preprocess"]
        text_features = bundle["text_features"]
        owners = bundle["owners"]
        device = bundle["device"]

        cols = st.columns(4)
        rows = []
        for i, path in enumerate(sample):
            try:
                img_t = load_image(path, preprocess, device)
                scores = score_image(model, img_t, text_features, owners, agg=agg)
                chosen = decide_label(scores, decision, threshold, margin, ratio, topk)
                year, month = get_exif_year_month(path)
                target = dst / chosen[0] / f"{year:04d}" / f"{month:02d}" / path.name
                rows.append((path.name, chosen[0], max(scores.items(), key=lambda kv: kv[1])[1], str(target)))

                with cols[i % 4]:
                    st.image(str(path), use_container_width=True)
                    st.caption(f"{path.name}\n→ {chosen[0]} • {year}/{month:02d}")
            except Exception as e:
                rows.append((path.name, "_error_", 0.0, f"error: {e}"))
        st.subheader("Planned actions (preview)")
        import pandas as pd
        df = pd.DataFrame(rows, columns=["file", "label", "top_score", "target_path"])
        st.dataframe(df, hide_index=True, use_container_width=True)


# FULL RUN -------------------------------------------------------------------

def build_cli_command():
    exe = [sys.executable, "photo_sorter.py",
           "--src", str(src),
           "--dst", str(dst),
           "--labels", str(labels_path),
           "--agg", agg,
           "--decision", decision,
           "--topk", str(topk),
           "--threshold", str(threshold),
           "--margin", str(margin),
           "--ratio", str(ratio)]
    if copy:
        exe.append("--copy")
    if dedupe:
        exe.append("--dedupe")
    if rich_prompts:
        exe.append("--rich-prompts")
    if ignore_weights:
        exe.append("--ignore-weights")

    # New flags
    if no_overwrite:
        exe.append("--no-overwrite")
    if skip_existing:
        exe.append("--skip-existing")
    if scan_dst:
        exe.append("--scan-dst-dedupe")
    return exe


def run_cli(exe: List[str]):
    st.write("Running:")
    st.code(" ".join([f'\"{x}\"' if " " in x else x for x in exe]), language="bash")
    with st.status("Executing full run…", expanded=True) as status:
        proc = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        CHILD_PROCS.append(proc)
        for line in proc.stdout:  # type: ignore
            st.write(line.rstrip())
        code = proc.wait()
        if proc in CHILD_PROCS:
            CHILD_PROCS.remove(proc)
        if code == 0:
            status.update(label="Done", state="complete")
            st.success("Full run completed successfully.")
        else:
            status.update(label="Failed", state="error")
            st.error(f"Process exited with code {code}")


if run_copy or run_move:
    if not src.exists() or not labels_path.exists():
        st.error("Please fix source/labels paths before running.")
    else:
        exe = build_cli_command()
        if run_move:
            exe = [x for x in exe if x != "--copy"]
        run_cli(exe)