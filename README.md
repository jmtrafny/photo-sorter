# Photo Organizer

Local, zero‑API photo organizer that classifies images with OpenCLIP and routes them into category folders and `YYYY/MM` subfolders pulled from EXIF. Includes a Streamlit UI with a safe auto‑shutdown (EXE‑friendly) and a dry‑run workflow.

> Runs entirely on your machine (CPU or CUDA if available). Supported images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif`. HEIC is optional (see **HEIC support**).

---

## Features

* **Zero‑shot tagging** via OpenCLIP (ViT‑B/32) using editable prompts in `labels.json`.
* **Date routing** to `<dst>/<category>/<YYYY>/<MM>/` using EXIF `DateTimeOriginal` (falls back to file mtime).
* **Dry‑run** mode to preview all actions safely.
* Optional **perceptual de‑duplication** (average‑hash) within a run.
* **Streamlit UI** (preview + full run) with an **auto‑shutdown** toggle so the packaged EXE quits when all tabs close.

---

## What’s in this repo

* `photo_sorter.py` – CLI that does the sorting.
* `app_streamlit.py` – Streamlit UI (preview + full run) with auto‑shutdown.
* `labels.json` – Config of categories, synonyms, and optional weights.
* (Docs in Canvas) **Run & Tuning Notes**, **Destination Folder Behavior**.

---

## Installation

Requirements: Python 3.10+ recommended; Windows/macOS/Linux.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Typical `requirements.txt` (yours may already exist):

```
open_clip_torch
torch  # or torch+cuda for your GPU build
pillow
imagehash
streamlit
websockets>=12.0   # only needed by the WebSocket UI flavor
tqdm
pillow-heif        # optional: HEIC support
```

> If you don’t have a GPU build of PyTorch, CPU is fine (just slower).

---

## Quick start (CLI)

**Safe dry‑run** (prints what *would* happen; no files moved):

```powershell
python photo_sorter.py `
  --src "C:\Photos\Unsorted" `
  --dst "C:\Photos\Sorted" `
  --agg max --decision margin --margin 0.008 `
  --ignore-weights --rich-prompts `
  --dry-run --debug-scores --debug-topk 5
```

**First real pass (copy instead of move):**

```powershell
python photo_sorter.py `
  --src "C:\Photos\Unsorted" `
  --dst "C:\Photos\Sorted" `
  --agg max --decision margin --margin 0.008 `
  --ignore-weights --rich-prompts `
  --dedupe --topk 1 --copy
```

Remove `--copy` when you’re happy to perform moves.

> Scores are normalized across labels. Typical top‑1 values are \~`0.100–0.110`. Margin/ratio rules are steadier than a high absolute threshold.

**Recommended defaults**

```
--agg max --decision margin --margin 0.008 --ignore-weights --rich-prompts
```

Looser/stricter:

* Looser: `--margin 0.006`
* Stricter: `--margin 0.012`
* Alternative: `--decision ratio --ratio 1.08`

---

## Streamlit UI

Preview sample images with the same knobs as the CLI and optionally run the full sort.

```bash
streamlit run app_streamlit.py
```

**Sidebar controls**

* **Paths**: `--src`, `--dst`, and `--labels` file.
* **Routing**: `--agg`, `--decision`, `--margin` / `--ratio` / `--threshold`.
* **Behavior**: `--topk`, `--copy`, `--dedupe`, `--rich-prompts`, `--ignore-weights`.
* **Lifecycle**: *Auto‑stop app when all tabs close* (toggle) and **Shutdown app now** button.

**How auto‑shutdown works**

* The app keeps a tiny per‑tab heartbeat. When **all tabs disconnect** for a short grace period, the process exits and any spawned child processes are cleaned up.
* This is EXE‑friendly (no orphan background processes). The toggle in the sidebar enables/disables it.

**UI workflow**

1. Pick a sample size and click **Preview** (dry) to see predicted destinations for a subset of images.
2. Inspect thumbnails and the **Planned actions** table.
3. Kick off **Run FULL (COPY)** for a safe first run, or **Run FULL (MOVE)** once you’re confident.

---

## CLI reference

### Required

* `--src <path>` – Folder containing unsorted images.
* `--dst <path>` – Destination root for sorted images.

### Labels & prompts

* `--labels <path>` *(default: `labels.json`)* – Label config.
* `--rich-prompts` – Use multiple prompt templates per synonym (slightly slower, better separation).
* `--ignore-weights` – Treat all label weights as `1.0`.

### Scoring & aggregation

* `--agg {max,mean,sum}` *(default: `max`)* – Aggregate per‑label scores from its synonym prompts.

  * `max` avoids bias toward labels with many synonyms (recommended).

### Decision policies

* `--decision {threshold,margin,ratio,always-top1}` *(default: `margin`)*

  * `threshold`: accept top‑1 if `top1 >= --threshold`.
  * `margin`: accept top‑1 if `(top1 - top2) >= --margin`.
  * `ratio`: accept top‑1 if `(top1 / max(top2,1e-9)) >= --ratio`.
  * `always-top1`: always route to the top label.
* `--threshold <float>` *(default: `0.10`)*
* `--margin <float>` *(default: `0.008`)* – good range `0.006–0.012`.
* `--ratio <float>` *(default: `1.06`)* – try `1.06–1.12`.

### Copy/move & duplicates

* `--copy` – Copy instead of move (safer first run).
* `--topk <int>` *(default: `1`)* – With `--copy`, you can duplicate into the top‑K categories.
* `--dedupe` – Per‑run perceptual de‑duplication (average‑hash). Duplicates go to `<dst>/_duplicates/`.

### Debugging & safety

* `--dry-run` – Print planned actions without changing files.
* `--debug-scores` – Print top label scores per image.
* `--debug-topk <int>` *(default: `5`)* – How many labels to print with `--debug-scores`.
* `--debug-prompts` – Also print the top matching prompt/synonym hits (slower).

---

## Destination behavior

* **Directories** – Existing `<category>/<YYYY>/<MM>` folders are **reused**; the code uses `mkdir(..., exist_ok=True)`.
* **MOVE (default)** – Uses a safe move that **never overwrites**. If a name exists, it writes a new one like `photo__1.jpg`, `photo__2.jpg`, ...
* **COPY (`--copy`)** – Mirrors the filename and **will overwrite** an existing file of the same name (no uniqueness check).
* **Duplicates vs prior runs** – `--dedupe` compares **within this run’s source set** only; it does **not** scan `<dst>`.

**Planned flags** (can be added quickly if you want):

* `--no-overwrite` (for copy paths) – copy with unique names (mirror of safe move).
* `--skip-existing` – skip a file if the destination exists.
* `--scan-dst-dedupe` – optional pre‑scan of `<dst>` to build a hash index and skip cross‑run duplicates.

---

## Folder structure

Each file is routed to:

```
<dst>/<category>/<YYYY>/<MM>/<original_filename>
```

Low‑confidence images (or when the policy rejects the top label) go to:

```
<dst>/_unsorted/<YYYY>/<MM>/
```

---

## Editing `labels.json`

```json
{
  "people": {
    "prompt": "people",
    "synonyms": ["people", "person", "portrait", "selfie"],
    "weight": 1.0
  }
}
```

Tips:

* Keep synonyms focused; avoid overly broad terms that bleed across categories.
* If a label dominates, either reduce its `weight` or run with `--ignore-weights` (recommended default).

---

## HEIC support

Install Pillow‑HEIF and enable it before using the CLI/UI to preview HEIC:

```bash
pip install pillow-heif
```

(You may also need platform codecs on Windows.)

---

## Performance tips

* **GPU**: If CUDA is available, PyTorch will pick it automatically.
* **CPU**: Expect slower runs. Consider disabling `--rich-prompts` or using smaller batches of files.
* Use `--agg max` and keep a concise set of synonyms.

---

## Troubleshooting

* `UserWarning: QuickGELU mismatch … 'openai'` from `open_clip` – benign; safe to ignore.
* Everything routes to one label – use `--agg max`, add `--ignore-weights`, enable `--rich-prompts`, and tighten `--margin`.
* Too many `_unsorted` – loosen `--margin` (e.g., `0.006`) or try `--decision always-top1` for a pass that never rejects.
* UI/EXE won’t close – ensure the **Auto‑stop** toggle is on; the app exits a few seconds after the last tab closes.

---

## Safety workflow

1. **Preview a sample** — Run a small **dry‑run** (or use the UI “Preview”) to see where files would go.
2. **Tweak the knobs** — Adjust `--decision` and `--margin`/`--ratio`, and/or edit `labels.json`.
3. **Repeat 1–2** until the sample looks right (top‑1 categories make sense and `_unsorted` rate is acceptable).
4. **Full pass (choose one)**

   * **Move originals**: run **without** `--copy`. Files are moved out of your unsorted folder.
   * **Keep originals**: run **with** `--copy`. Originals stay put; sorted copies are written to `--dst`.

> Tip: If you do a full pass with `--copy` and later decide to run a **move**, point `--dst` to an empty folder **or** use safeguards (e.g., `--skip-existing` / `--no-overwrite`) to avoid clobbering the previous copies.

---

## FAQ

**Does it modify originals?** Only in **MOVE** mode. Use `--copy` for a non‑destructive first pass.

**Can a photo live in multiple categories?** Yes: `--topk > 1` with `--copy` duplicates it across the top‑K labels.

**Will it find duplicates already in `<dst>`?** Not yet. Current `--dedupe` is in‑run only; see planned flags above.

---

## Roadmap

* Optional `--no-overwrite`, `--skip-existing`, `--scan-dst-dedupe`.
* Smarter heuristics for `_unsorted` routing.
* HEIC improvement and video (`.mp4/.mov`) stubs.

---

## License

MIT License — see the bottom of this README.

```
Copyright (c) 2025 James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgements

* [OpenCLIP](https://github.com/mlfoundations/open_clip) for CLIP models & utilities.
* PyTorch, Pillow, ImageHash, Streamlit.
