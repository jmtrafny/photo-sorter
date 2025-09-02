# Photo Organizer — AI-Powered Photo Sorting

Local, privacy-focused photo organizer that:

* Generates **content tags** with OpenCLIP models (4 sizes: small/medium/large/xlarge) using zero‑shot classification
* **Smart caching** — models download once and cache locally for future use
* **Tiered label sets** — Small (8), Medium (16), Large (40+) categories for different accuracy needs  
* Sorts into **category folders** and **Year/Month** subfolders from EXIF
* **Batch processing** with parallel image loading for better performance
* Optional **duplicate detection** (perceptual hash)
* **Preview mode** to see moves before execution
* **Intuitive Tkinter GUI** with model selection and processing time estimates

> Runs on CPU or GPU (CUDA if available). Supported image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif`. HEIC can be added later.

## Quick Start - GUI Application

**Download and run** the standalone executable:

1. Download `PhotoOrganizer.exe` from releases
2. Double-click to launch the GUI
3. Select source and destination folders
4. Click "Preview" to see where photos will be sorted
5. Review the preview images and operations table
6. Click "Run Full Sort" when ready

No Python installation required!

- - -

## Developer Installation

If you want to run from source or contribute to development:

``` bash
python -m venv .venv

# Windows: 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate

# macOS/Linux: 
source .venv/bin/activate

pip install -r requirements.txt
```

**Run from source:**
``` bash
# GUI version
python app_tkinter.py

# CLI version  
python photo_sorter.py --help
```

**Build executable:**
``` bash
python build_exe.py
```

- - -

## CLI Usage (Advanced)

The command-line interface provides more granular control:

**Safe preview** (shows what would happen, no files moved):

``` powershell
python photo_sorter.py `
  --src "path\to\Unsorted" `
  --dst "path\to\Sorted" `
  --model-size medium --label-tier medium `
  --agg max --decision margin --margin 0.008 `
  --ignore-weights --rich-prompts `
  --dry-run --debug-scores --debug-topk 5
```

**First real pass (copy instead of move):**

``` powershell
python photo_sorter.py `
  --src "path\to\Unsorted" `
  --dst "path\to\Sorted" `
  --model-size medium --label-tier medium `
  --agg max --decision margin --margin 0.008 `
  --ignore-weights --rich-prompts `
  --dedupe --topk 1 --copy
```

Remove `--copy` to actually move files once you're happy.

- - -

## Routing & Tuning Notes

Scores are **normalized across labels**, so the top score typically sits around **0.100–0.110**. Margin/ratio rules are therefore more stable than a high absolute threshold.

**Recommended defaults:**

``` powershell
--model-size medium --label-tier medium --agg max --decision margin --margin 0.008 --ignore-weights --rich-prompts
```

Adjust confidence:

* Looser: `--margin 0.006`
* Stricter: `--margin 0.012`
* Alternative: `--decision ratio --ratio 1.08`

- - -

## CLI reference (all flags)

### Required

* `--src <path>`: Folder containing unsorted images.
* `--dst <path>`: Destination root for sorted images.

### Model & labels

* `--model-size {small,medium,large,xlarge}` *(default: `small`)*: AI model size - small (fast), medium (balanced), large (accurate), xlarge (max accuracy).
* `--label-tier {small,medium,large}` *(auto-detect if not specified)*: Label complexity - small (8 labels), medium (16 labels), large (40+ labels). Auto-selects based on model size if not specified.
* `--labels <path>` *(optional)*: Custom label config JSON file. Uses built-in tiered labels if not provided.
* `--rich-prompts` *(bool)*: Use multiple prompt templates per synonym for better separation (slightly slower).
* `--ignore-weights` *(bool)*: Treat all label weights as 1.0 to avoid category bias.

### Scoring & aggregation

* `--agg {max,mean,sum}` *(default: <b>**************************************************`max`**************************************************</b>)*: Aggregate per‑label scores from its synonym prompts.
    * `max` resists synonym‑count bias (recommended).
    * `mean` balances across synonyms.
    * `sum` can bias toward labels with many synonyms.

### Decision policies (how we accept/route a label)

* `--decision {threshold,margin,ratio,always-top1}` *(default: <b>**************************************************`margin`**************************************************</b>)*
    * `threshold`: accept top‑1 if `top1 >= --threshold`.
    * `margin`: accept top‑1 if `(top1 - top2) >= --margin`.
    * `ratio`: accept top‑1 if `(top1 / max(top2,1e-9)) >= --ratio`.
    * `always-top1`: always route to the top label (no rejection).

**Policy parameters**

* `--threshold <float>` *(default: <b>**************************************************`0.10`**************************************************</b>)*: Used by `decision=threshold`.
* `--margin <float>` *(default: <b>**************************************************`0.008`**************************************************</b>)*: Used by `decision=margin`. Good starting range: `0.006–0.012`.
* `--ratio <float>` *(default: <b>**************************************************`1.06`**************************************************</b>)*: Used by `decision=ratio`. Try `1.06–1.12`.

### Copy/move behavior

* `--copy` *(bool)*: Copy files instead of moving (safer first run).
* `--topk <int>` *(default: <b>**************************************************`1`**************************************************</b>)*: Route to top‑K labels; with `--copy` you'll get duplicates across categories.
* `--date-folders` *(bool)*: Organize photos into YYYY/MM subfolders based on EXIF or file date. Without this flag, photos are placed directly in label folders.

### Duplicates

* `--dedupe` *(bool)*: Perceptual duplicate detection via average‑hash. Duplicates are placed in `dst/_duplicates/`.

### Performance

* `--batch-size <int>` *(auto-detect if not specified)*: Batch size for processing images. Auto-detects optimal size based on model and available memory.
* `--parallel-load <int>` *(default: number of CPU cores)*: Number of parallel workers for image loading to improve performance.

### Debugging & safety

* `--dry-run` *(bool)*: Show planned actions without changing files.
* `--debug-scores` *(bool)*: Print top label scores per image.
* `--debug-topk <int>` *(default: `5`)*: How many labels to print with `--debug-scores`.
* `--debug-prompts` *(bool)*: Print top matching prompt/synonym hits (slower; use on small test sets).
* `--skip-existing` *(bool)*: Skip files if destination already exists (useful for incremental runs).
* `--no-overwrite` *(bool)*: Error and stop if destination exists instead of auto-renaming (safety guard).

- - -

## Folder structure

**With `--date-folders` flag:**
```
<dst>/<category>/<YYYY>/<MM>/<original_filename>
```

**Without `--date-folders` flag (default):**
```
<dst>/<category>/<original_filename>
```

* When using date folders, EXIF `DateTimeOriginal` is preferred; falls back to file modified time.
* Low‑confidence images (or when policy rejects the top label) go to `<dst>/_unsorted/` (or `<dst>/_unsorted/<YYYY>/<MM>/` with date folders).

- - -

## Model Selection & Performance

### Choosing Model Size

The application supports 4 different model sizes with different accuracy/speed tradeoffs:

* **Small** (ViT-B-32): ~60% accuracy, fastest processing, ~150MB download
* **Medium** (ViT-B-16): ~75% accuracy, moderate speed, ~300MB download  
* **Large** (ViT-L-14): ~85% accuracy, slower processing, ~900MB download
* **XLarge** (ViT-L-14-336): ~90% accuracy, slowest but most accurate, ~900MB download

### Smart Caching

* Models are **automatically cached** after first download to your system's app data folder
* **No repeated downloads** - subsequent runs load instantly from cache
* Works without admin privileges or developer mode (uses degraded symlink mode on Windows)
* Cache location: `%APPDATA%\photo_organizer` on Windows, `~/.cache/photo_organizer` on Linux/Mac

### Label Tiers

Choose label complexity based on your needs:

* **Small (8 labels)**: people, pets, landscape, food, documents, cars, buildings, events
* **Medium (16 labels)**: Adds wildlife, cityscape, screenshots, flowers, kids, sports, art, indoor, selfies  
* **Large (40+ labels)**: Comprehensive set including portraits, family, mountains, beach, architecture, dessert, wedding, etc.

If not specified, label tier auto-selects based on model size.

- - -

## Editing labels

Open `labels.json` and edit or add categories:

``` json
{
  "people": {
    "prompt": "people",
    "synonyms": ["people", "person", "portrait", "selfie"],
    "weight": 1.0
  }
}
```

Tips:

* Keep synonyms focused; avoid overly broad terms that overlap many classes.
* If a label still dominates, either reduce its `weight` or run with `--ignore-weights` (recommended default).

- - -

## Troubleshooting

* **OpenCLIP warning** `QuickGELU mismatch … 'openai'`: benign; safe to ignore.
* **Everything routes to one label**: use `--agg max`, add `--ignore-weights`, enable `--rich-prompts`.
* \*\*Too many\*\* `_unsorted`: try looser `--margin` (e.g., `0.006`) or use `--decision always-top1` for a pass that never rejects.
* **HEIC images**: not supported by default. Coming soon!

- - -

## Safety workflow

### GUI Workflow (Recommended)

1. **Launch the app** — Double-click `PhotoOrganizer.exe`
2. **Select folders** — Choose source (unsorted) and destination folders
3. **Preview** — Click "Preview (Dry Run)" to see where photos will be sorted
4. **Review** — Check the preview images and operations summary table
5. **Adjust settings** — Modify labels, thresholds, or other options if needed
6. **Execute** — Click "Run Full Sort" to perform the actual organization

### CLI Workflow (Advanced)

1. **Preview a sample** — Run a small **dry‑run** to see where files would go.
2. **Tweak the knobs** — Adjust `--decision` and `--margin`/`--ratio`, and/or edit `labels.json`.
3. **Repeat 1–2** until the sample looks right (top‑1 categories make sense and `_unsorted` rate is acceptable).
4. **Full pass (choose one)**

   * **Move originals**: run **without** `--copy`. Files are moved out of your unsorted folder.
   * **Keep originals**: run **with** `--copy`. Originals stay put; sorted copies are written to `--dst`.

> Tip: If you do a full pass with `--copy` and later decide to run a **move**, point `--dst` to an empty folder **or** use safeguards (`--skip-existing` / `--no-overwrite`) to avoid clobbering the previous copies.

- - -

## License

MIT License

Copyright (c) 2025 James M. Trafny

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