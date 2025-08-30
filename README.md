# Photo Organizer — Weekend MVP

Local, free-ish (no paid APIs) photo organizer that:

* Generates **content tags** with OpenCLIP (ViT‑B/32) zero‑shot prompts from `labels.json`.
* Sorts into **category folders** and **Year/Month** subfolders from EXIF.
* Optional **duplicate detection** (perceptual hash).
* **Dry‑run** to preview moves.

> Runs on CPU or GPU (CUDA if available). Supported image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif`. HEIC can be added later.

- - -

## Install

``` bash
python -m venv .venv

# Windows: 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate

# macOS/Linux: 
source .venv/bin/activate

pip install -r requirements.txt
```

- - -

## Quick start

**Safe dry‑run** (prints what would happen, no files moved):

``` powershell
python photo_sorter.py `
  --src "path\to\Unsorted" `
  --dst "path\to\Sorted" `
  --agg max --decision margin --margin 0.008 `
  --ignore-weights --rich-prompts `
  --dry-run --debug-scores --debug-topk 5
```

**First real pass (copy instead of move):**

``` powershell
python photo_sorter.py `
  --src "path\to\Unsorted" `
  --dst "path\to\Sorted" `
  --agg max --decision margin --margin 0.008 `
  --ignore-weights --rich-prompts `
  --dedupe --topk 1 --copy
```

Remove `--copy` to actually move files once you’re happy.

- - -

## Routing & Tuning Notes

Scores are **normalized across labels**, so the top score typically sits around **0.100–0.110**. Margin/ratio rules are therefore more stable than a high absolute threshold.

**Recommended defaults:**

``` powershell
--agg max --decision margin --margin 0.008 --ignore-weights --rich-prompts
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

### Labels & prompts

* `--labels <path>` *(default: <b>**************************************************`labels.json`**************************************************</b>)*: Label config with `prompt`, `synonyms`, `weight`.
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
* `--topk <int>` *(default: <b>**************************************************`1`**************************************************</b>)*: Route to top‑K labels; with `--copy` you’ll get duplicates across categories.

### Duplicates

* `--dedupe` *(bool)*: Perceptual duplicate detection via average‑hash. Duplicates are placed in `dst/_duplicates/`.

### Debugging & safety

* `--dry-run` *(bool)*: Show planned actions without changing files.
* `--debug-scores` *(bool)*: Print top label scores per image.
* `--debug-topk <int>` *(default: <b>**************************************************`5`**************************************************</b>)*: How many labels to print with `--debug-scores`.
* `--debug-prompts` *(bool)*: Print top matching prompt/synonym hits (slower; use on small test sets).

- - -

## Folder structure

Each file is routed to:

```
<dst>/<category>/<YYYY>/<MM>/<original_filename>
```

* EXIF `DateTimeOriginal` is preferred; falls back to file modified time.
* Low‑confidence images (or when policy rejects the top label) go to `<dst>/_unsorted/<YYYY>/<MM>/`.

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

1. **Preview a sample** — Run a small **dry‑run** (or use the UI “Preview”) to see where files would go.
2. **Tweak the knobs** — Adjust `--decision` and `--margin`/`--ratio`, and/or edit `labels.json`.
3. **Repeat 1–2** until the sample looks right (top‑1 categories make sense and `_unsorted` rate is acceptable).
4. **Full pass (choose one)**

   * **Move originals**: run **without** `--copy`. Files are moved out of your unsorted folder.
   * **Keep originals**: run **with** `--copy`. Originals stay put; sorted copies are written to `--dst`.

> Tip: If you do a full pass with `--copy` and later decide to run a **move**, point `--dst` to an empty folder **or** use safeguards (e.g., `--skip-existing` / `--no-overwrite`) to avoid clobbering the previous copies.

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