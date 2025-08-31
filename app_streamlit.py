#!/usr/bin/env python3
r"""
Streamlit UI for Photo Organizer
Run: `streamlit run app_streamlit.py`
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
import warnings

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ExifTags
import torch
import open_clip

from default_labels import DEFAULT_LABELS

# Silence the benign open_clip warning about QuickGELU
warnings.filterwarnings(
    "ignore", 
    message="QuickGELU mismatch.*",
    category=UserWarning,
    module="open_clip.factory"
)

try:
    import websockets
    # Prefer legacy path to avoid DeprecationWarning from websockets>=12
    try:  # websockets>=12
        from websockets.legacy.server import WebSocketServerProtocol
    except Exception:  # fallback for older versions
        from websockets.server import WebSocketServerProtocol
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "websockets is required. Please add `websockets>=12.0` to requirements.txt"
    ) from e

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
CHILD_PROCS: List[subprocess.Popen] = []

_WS_PORT: int = 0
_WS_SERVER = None
_WS_LOOP: asyncio.AbstractEventLoop | None = None
_WS_CLIENTS: set[WebSocketServerProtocol] = set()
_WS_LOCK = threading.Lock()
_SEEN_A_CLIENT = False
_AUTO_SHUTDOWN_ENABLED = True
_GRACE_SECONDS = 60.0


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
    _terminate_children()
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


async def _ws_handle_client(ws: WebSocketServerProtocol):
    global _SEEN_A_CLIENT
    with _WS_LOCK:
        _WS_CLIENTS.add(ws)
        print(f"[WS] client connected; total={len(_WS_CLIENTS)}")
    _SEEN_A_CLIENT = True
    try:
        async for _ in ws:
            await ws.send("pong")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with _WS_LOCK:
            _WS_CLIENTS.discard(ws)
            print(f"[WS] client disconnected; total={len(_WS_CLIENTS)}")


async def _ws_main(port: int):
    async def _runner(port: int):
        async with websockets.serve(
            _ws_handle_client,
            host="0.0.0.0",  # accept localhost or LAN host
            port=port,
            ping_interval=2,
            ping_timeout=4,
            origins=None,
        ):
            await asyncio.Future()
    return _runner(port)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))  # find a port usable from localhost or LAN IP
        return s.getsockname()[1]


@st.cache_resource(show_spinner=False)
def _start_ws_server() -> int:
    global _WS_LOOP
    port = _find_free_port()
    loop = asyncio.new_event_loop()
    _WS_LOOP = loop

    async def _start():
        async with websockets.serve(
            _ws_handle_client,
            host="0.0.0.0",
            port=port,
            ping_interval=2,
            ping_timeout=4,
            origins=None,
        ):
            await asyncio.Future()

    def _runner():
        try:
            loop.run_until_complete(_start())
        except Exception:
            pass

    threading.Thread(target=_runner, daemon=True, name="ws-server").start()
    return port


def _start_ws_watcher():
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
                pass

    threading.Thread(target=_watch, daemon=True, name="ws-watcher").start()


def _inject_ws_client(port: int):
    # Inject a hidden script-only client; no visible UI elements.
    html_code = f"""
    <script>
    (function() {{
      if (window.__photo_ws_ready__) return; // guard per tab
      window.__photo_ws_ready__ = true;
      const url = 'ws://127.0.0.1:{port}';
      let ws = null;
      let pingTimer = null;
      function connect() {{
        try {{
          ws = new WebSocket(url);
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
            if (!document.hidden) {{ setTimeout(connect, 1000); }}
          }};
          ws.onerror = function(_e) {{ try {{ ws.close(); }} catch (_err) {{}} }};
          window.addEventListener('pagehide', function() {{ try {{ ws.close(); }} catch (_e) {{}} }}, {{ passive: true }});
          window.addEventListener('beforeunload', function() {{ try {{ ws.close(); }} catch (_e) {{}} }}, {{ passive: true }});
        }} catch (_e) {{ /* swallow */ }}
      }}
      connect();
    }})();
    </script>
    """
    components.html(html_code, height=0)


# ----------------------------
# Helpers: Directory picker via Tkinter
# ----------------------------

def pick_directory(initial: str | Path | None = None) -> str | None:
    """Open a native folder chooser and return the selected path or None.
    Falls back gracefully if Tkinter is unavailable (e.g., headless env).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        # keep the dialog on top of other windows
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        initialdir = str(initial) if initial else None
        path = filedialog.askdirectory(initialdir=initialdir, title="Select folder")
        try:
            root.destroy()
        except Exception:
            pass
        return path or None
    except Exception as e:
        # If something goes wrong, just return None (user can type path)
        return None


# ----------------------------
# Shared logic (mirrors CLI)
# ----------------------------

def validate_labels(data: Dict[str, Dict]) -> Tuple[bool, str]:
    """Validate labels configuration structure."""
    if not isinstance(data, dict):
        return False, "Labels must be a JSON object/dictionary"
    
    if not data:
        return False, "Labels cannot be empty"
    
    for label, cfg in data.items():
        if not isinstance(cfg, dict):
            return False, f"Label '{label}' must be an object with 'prompt' and 'synonyms'"
        
        if "synonyms" in cfg and not isinstance(cfg["synonyms"], list):
            return False, f"Label '{label}': 'synonyms' must be an array"
        
        if "weight" in cfg:
            try:
                float(cfg["weight"])
            except (TypeError, ValueError):
                return False, f"Label '{label}': 'weight' must be a number"
    
    return True, ""


def load_labels(labels_path: Path | None = None, use_builtin: bool = True) -> Tuple[Dict[str, Dict], str]:
    """Load labels with validation. Returns (labels_dict, status_message)"""
    data = None
    status = ""
    
    if labels_path and labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            is_valid, error_msg = validate_labels(data)
            if not is_valid:
                if use_builtin:
                    status = f"⚠️ Custom labels malformed: {error_msg}. Using built-in labels."
                    data = None
                else:
                    raise ValueError(f"Invalid labels: {error_msg}")
            else:
                status = f"✅ Using custom labels from {labels_path.name}"
        except json.JSONDecodeError as e:
            if use_builtin:
                status = f"⚠️ Failed to parse labels file: {e}. Using built-in labels."
                data = None
            else:
                raise
    
    if data is None:
        data = DEFAULT_LABELS.copy()
        if not status:
            status = "✅ Using built-in labels"
    
    # Ensure proper structure
    for label, cfg in data.items():
        cfg.setdefault("synonyms", [])
        base_prompt = cfg.get("prompt", label)
        if not cfg["synonyms"] or base_prompt not in cfg["synonyms"]:
            cfg["synonyms"].insert(0, base_prompt)
        cfg.setdefault("weight", 1.0)
    
    return data, status


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


# --- Page ---
st.set_page_config(page_title="Photo Organizer — Preview & Run", layout="wide")
st.title("📷 Photo Organizer — Preview & Full Run")

_WS_PORT = _start_ws_server()
_start_ws_watcher()
_inject_ws_client(_WS_PORT)

# Seed session defaults for text inputs (so Browse… can update them)
if "src_str" not in st.session_state:
    st.session_state.src_str = str(Path.cwd())
if "dst_str" not in st.session_state:
    st.session_state.dst_str = str(Path.cwd() / "sorted")
if "labels_str" not in st.session_state:
    st.session_state.labels_str = str(Path("labels.json").resolve())

with st.sidebar:
    st.header("Settings")

    # Source selector with integrated file chooser
    st.markdown("**Source folder (--src)**")
    src_placeholder = st.empty()
    with src_placeholder.container():
        if st.button(f"📁 {st.session_state.src_str}", key="src_chooser", use_container_width=True, 
                     help="Click to browse for source folder"):
            picked = pick_directory(st.session_state.src_str)
            if picked:
                st.session_state.src_str = picked
                st.rerun()

    # Destination selector with integrated file chooser  
    st.markdown("**Destination (--dst)**")
    dst_placeholder = st.empty()
    with dst_placeholder.container():
        if st.button(f"📁 {st.session_state.dst_str}", key="dst_chooser", use_container_width=True,
                     help="Click to browse for destination folder"):
            picked = pick_directory(st.session_state.dst_str)
            if picked:
                st.session_state.dst_str = picked
                st.rerun()

    # Labels configuration selector
    st.markdown("**Labels Configuration**")
    use_custom = st.checkbox("Use custom labels file", value=False,
                             help="Check to use a custom labels.json file instead of built-in labels")
    
    if use_custom:
        st.session_state.labels_str = st.text_input("Custom labels path", 
                                                    value=st.session_state.get("labels_str", "labels.json"),
                                                    help="Path to custom JSON file defining categories")
        labels_path = Path(st.session_state.labels_str)
    else:
        labels_path = None
        st.info("Using built-in labels (people, pets, landscape, food, etc.)")

    st.subheader("Routing")
    agg = st.selectbox("Aggregation (--agg)", ["max", "mean", "sum"], index=0,
                       help="How to combine scores from multiple synonyms. 'max' prevents bias from labels with many synonyms")
    decision = st.selectbox("Decision (--decision)", ["margin", "ratio", "threshold", "always-top1"], index=0,
                            help="Rule for accepting a label: margin (top1-top2 gap), ratio (top1/top2), threshold (minimum score), or always-top1 (never reject)")
    margin = st.slider("Margin (--margin)", 0.0, 0.03, 0.008, 0.001,
                       help="Minimum difference between top two scores (used with margin decision). Higher = stricter")
    ratio = st.slider("Ratio (--ratio)", 1.00, 1.30, 1.06, 0.01,
                      help="Minimum ratio of top1/top2 scores (used with ratio decision). Higher = stricter")
    threshold = st.slider("Threshold (--threshold)", 0.05, 0.4, 0.10, 0.01,
                          help="Minimum confidence score to accept a label (used with threshold decision). Higher = stricter")

    st.subheader("Behavior")
    topk = st.selectbox("Top-K (--topk)", [1, 2, 3], index=0,
                        help="Number of top categories to sort into. With copy mode, creates duplicates in multiple folders")
    copy = st.checkbox("Copy (not move) (--copy)", value=False,
                       help="Copy files instead of moving them. Originals stay in source folder")
    dedupe = st.checkbox("Deduplicate (--dedupe)", value=False,
                         help="Detect near-duplicates using perceptual hashing and move them to _duplicates folder")
    rich_prompts = st.checkbox("Rich prompts (--rich-prompts)", value=True,
                                help="Use multiple prompt templates per synonym for better accuracy (slightly slower)")
    ignore_weights = st.checkbox("Ignore weights (--ignore-weights)", value=True,
                                  help="Treat all labels equally (weight=1.0) to prevent category bias")
    skip_existing = st.checkbox("Skip existing (--skip-existing)", value=False, 
                                 help="Skip files if destination already exists (useful for incremental runs)")
    no_overwrite = st.checkbox("No overwrite (--no-overwrite)", value=False, 
                                help="Error if destination exists instead of auto-renaming (safety guard)")
    date_folders = st.checkbox("Date folders (--date-folders)", value=False,
                                help="Organize photos into YYYY/MM subfolders based on EXIF or file date")

    st.subheader("Preview sample")
    sample_n = st.number_input("Sample size", min_value=1, max_value=500, value=50, step=1)
    do_preview = st.button("🔎 Preview sample (dry)")

    st.subheader("Full run")
    run_copy = st.button("▶ Run FULL (COPY)")
    run_move = st.button("⚠ Run FULL (MOVE)")

# Materialize paths from session state
src = Path(st.session_state.src_str)
dst = Path(st.session_state.dst_str)

# Warnings for missing paths
if not src.exists():
    st.warning("Source folder does not exist.")

# Load and validate labels
if use_custom and st.session_state.get("labels_str"):
    custom_path = Path(st.session_state.labels_str)
    if not custom_path.exists():
        st.warning(f"Custom labels file not found: {custom_path}")
else:
    custom_path = None


@st.cache_resource(show_spinner=True)
def get_model_and_text_features(labels_path: Path | None, rich_prompts: bool, ignore_weights: bool, use_builtin: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    pretrained = "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    labels_cfg, status = load_labels(labels_path, use_builtin=use_builtin)
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
        "labels_status": status,
    }


@st.cache_data(show_spinner=False)
def list_images(folder: Path) -> List[Path]:
    imgs: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            imgs.append(p)
    return imgs


if do_preview and src.exists():
    imgs = list_images(src)
    if not imgs:
        st.info("No supported images found in source folder.")
    else:
        sample = imgs if len(imgs) <= sample_n else random.sample(imgs, sample_n)
        st.write(f"Previewing **{len(sample)}** of **{len(imgs)}** images from `{src}`.")

        bundle = get_model_and_text_features(labels_path if use_custom else None, rich_prompts, ignore_weights)
        if bundle["labels_status"]:
            st.info(bundle["labels_status"])
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
                
                if date_folders:
                    year, month = get_exif_year_month(path)
                    target = dst / chosen[0] / f"{year:04d}" / f"{month:02d}" / path.name
                    caption_text = f"{path.name}\n→ {chosen[0]} • {year}/{month:02d}"
                else:
                    target = dst / chosen[0] / path.name
                    caption_text = f"{path.name}\n→ {chosen[0]}"
                    
                rows.append((path.name, chosen[0], max(scores.items(), key=lambda kv: kv[1])[1], str(target)))

                with cols[i % 4]:
                    st.image(str(path), width='stretch')
                    st.caption(caption_text)
            except Exception as e:
                rows.append((path.name, "_error_", 0.0, f"error: {e}"))
        st.subheader("Planned actions (preview)")
        import pandas as pd
        df = pd.DataFrame(rows, columns=["file", "label", "top_score", "target_path"])
        st.dataframe(df, hide_index=True, width='stretch')


def build_cli_command():
    exe = [sys.executable, "photo_sorter.py",
           "--src", str(src),
           "--dst", str(dst)]
    
    # Add custom labels if specified
    if use_custom and labels_path:
        exe.extend(["--labels", str(labels_path)])
    
    exe.extend([
           "--agg", agg,
           "--decision", decision,
           "--topk", str(topk),
           "--threshold", str(threshold),
           "--margin", str(margin),
           "--ratio", str(ratio)])
    
    if copy:
        exe.append("--copy")
    if dedupe:
        exe.append("--dedupe")
    if rich_prompts:
        exe.append("--rich-prompts")
    if ignore_weights:
        exe.append("--ignore-weights")
    if skip_existing:
        exe.append("--skip-existing")
    if no_overwrite:
        exe.append("--no-overwrite")
    if date_folders:
        exe.append("--date-folders")
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
    if not src.exists():
        st.error("Please fix source path before running.")
    elif use_custom and labels_path and not labels_path.exists():
        st.error("Custom labels file not found. Please fix the path or use built-in labels.")
    else:
        exe = build_cli_command()
        if run_move:
            exe = [x for x in exe if x != "--copy"]
        run_cli(exe)
