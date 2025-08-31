#!/usr/bin/env python3
r"""
photo_sorter.py

Usage:
  python photo_sorter.py --src "C:\Photos\Unsorted" --dst "C:\Photos\Sorted" --dry-run
  python photo_sorter.py --src /path/in --dst /path/out --threshold 0.23 --dedupe
"""

import argparse
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

from PIL import Image, ExifTags
import imagehash
import torch
import open_clip
from tqdm import tqdm

from default_labels import DEFAULT_LABELS

# Silence the benign open_clip warning about QuickGELU
warnings.filterwarnings(
    "ignore",
    message="QuickGELU mismatch.*",
    category=UserWarning,
    module="open_clip.factory"
)

# ----------------------------
# Helpers
# ----------------------------

def validate_labels(data: Dict[str, Dict]) -> Tuple[bool, str]:
    """Validate labels configuration structure.
    Returns (is_valid, error_message)"""
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


def load_labels(labels_path: Path | None = None, use_builtin: bool = True) -> Dict[str, Dict]:
    """Load labels configuration with validation.
    
    Args:
        labels_path: Path to custom labels.json file (optional)
        use_builtin: If True and labels_path is None or invalid, use built-in labels
    
    Returns:
        Dictionary of label configurations
    """
    data = None
    
    # Try to load custom labels file if provided
    if labels_path and labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate the loaded data
            is_valid, error_msg = validate_labels(data)
            if not is_valid:
                print(f"[WARNING] Custom labels file is malformed: {error_msg}")
                if use_builtin:
                    print("[INFO] Falling back to built-in labels")
                    data = None
                else:
                    raise ValueError(f"Invalid labels file: {error_msg}")
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse labels file: {e}")
            if use_builtin:
                print("[INFO] Falling back to built-in labels")
                data = None
            else:
                raise
    
    # Use built-in labels if no custom file or if it failed
    if data is None:
        if use_builtin:
            data = DEFAULT_LABELS.copy()
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Ensure each label has proper structure
    for label, cfg in data.items():
        cfg.setdefault("synonyms", [])
        base_prompt = cfg.get("prompt", label)
        if not cfg["synonyms"] or base_prompt not in cfg["synonyms"]:
            cfg["synonyms"].insert(0, base_prompt)
        cfg.setdefault("weight", 1.0)
    
    return data


def _prompt_variants(syn: str, rich: bool) -> Iterable[str]:
    if not rich:
        yield f"a photo of {syn}"
        return
    # a small template set that often helps CLIP without exploding bias
    templates = [
        "{}",
        "a photo of {}",
        "an image of {}",
        "a close-up photo of {}",
        "a snapshot of {}",
    ]
    for t in templates:
        yield t.format(syn)


def build_text_tokens(model, tokenizer, labels_cfg: Dict[str, Dict], device: str, rich_prompts: bool, ignore_weights: bool):
    """Create the text embeddings for all label synonyms.
    Returns:
      text_features: (N_prompts, dim)
      owners: list of tuples (label, weight, synonym)
      texts: original prompt strings (for debug)
    """
    texts: List[str] = []
    owners: List[Tuple[str, float, str]] = []
    for label, cfg in labels_cfg.items():
        weight = 1.0 if ignore_weights else float(cfg.get("weight", 1.0))
        for syn in cfg.get("synonyms", []):
            for prompt_t in _prompt_variants(syn, rich_prompts):
                texts.append(prompt_t)
                owners.append((label, weight, syn))
    if not texts:
        raise RuntimeError("No text prompts generated from labels.json")
    text_tokens = tokenizer(texts)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens.to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, owners, texts


def load_image(path: Path, device: str, preprocess):
    img = Image.open(path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    return img_t


def score_labels(model, image_t, text_features, owners, *, agg: str = "max", temperature: float = 1.0):
    """Compute label scores by aggregating per-synonym probabilities.

    agg: one of {"max", "mean", "sum"}.  Default "max" avoids the bias where
         labels with many synonyms get inflated scores.
    """
    with torch.no_grad():
        image_features = model.encode_image(image_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) / temperature
        probs = logits.softmax(dim=-1).squeeze(0)

    # gather per-label values
    per_label_vals: Dict[str, List[float]] = {}
    for idx, p in enumerate(probs):
        label, weight, _syn = owners[idx]
        per_label_vals.setdefault(label, []).append(float(p) * float(weight))

    scores: Dict[str, float] = {}
    for label, vals in per_label_vals.items():
        if not vals:
            scores[label] = 0.0
        elif agg == "sum":
            scores[label] = sum(vals)
        elif agg == "mean":
            scores[label] = sum(vals) / len(vals)
        else:  # "max"
            scores[label] = max(vals)

    # normalize to sum=1 to make thresholding easier/comparable
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    return scores, probs


def get_exif_datetime(path: Path) -> Tuple[int, int]:
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
            dt = datetime.fromisoformat(parts)
            return dt.year, dt.month
    except Exception:
        pass
    dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt.year, dt.month


def safe_move(src: Path, dst: Path, dry_run: bool, skip_existing: bool = False, no_overwrite: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    
    if dst.exists():
        if skip_existing:
            print(f"[SKIP] {src} -> {dst} (destination exists)")
            return
        elif no_overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
    
    base, ext = dst.stem, dst.suffix
    candidate = dst
    i = 1
    while candidate.exists():
        candidate = dst.with_name(f"{base}__{i}{ext}")
        i += 1
    shutil.move(str(src), str(candidate))




def safe_copy(src: Path, dst: Path, dry_run: bool, skip_existing: bool = False, no_overwrite: bool = False):
    """Copy with collision-safe naming (appends __1, __2, ... if needed)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    
    if dst.exists():
        if skip_existing:
            print(f"[SKIP] {src} -> {dst} (destination exists)")
            return
        elif no_overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
    
    base, ext = dst.stem, dst.suffix
    candidate = dst
    i = 1
    while candidate.exists():
        candidate = dst.with_name(f"{base}__{i}{ext}")
        i += 1
    shutil.copy2(str(src), str(candidate))
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def compute_phash(path: Path):
    try:
        with Image.open(path) as im:
            return imagehash.average_hash(im)
    except Exception:
        return None


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Auto-sort photos by content using local zero-shot tagging (OpenCLIP).")
    ap.add_argument("--src", required=True, help="Folder with unsorted images")
    ap.add_argument("--dst", required=True, help="Folder to place sorted images")
    ap.add_argument("--labels", default=None, help="Path to custom labels.json (uses built-in labels if not provided)")
    ap.add_argument("--threshold", type=float, default=0.10, help="Confidence threshold when --decision=threshold (scores are normalized across labels)")
    ap.add_argument("--topk", type=int, default=1, help="Move into the highest scoring label (1) or copy into top-K labels (>1).")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    ap.add_argument("--dedupe", action="store_true", help="Detect and divert near-duplicates to a 'duplicates' folder (perceptual hash)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without moving files")
    ap.add_argument("--agg", choices=["max", "mean", "sum"], default="max", help="How to aggregate synonym scores per label (default: max)")
    ap.add_argument("--decision", choices=["threshold", "margin", "ratio", "always-top1"], default="margin", help="Routing rule: threshold/ratio/margin/always-top1")
    ap.add_argument("--margin", type=float, default=0.008, help="top1 - top2 must be >= margin when --decision=margin")
    ap.add_argument("--ratio", type=float, default=1.06, help="top1/top2 must be >= ratio when --decision=ratio")
    ap.add_argument("--debug-scores", action="store_true", help="Print top label scores for each image")
    ap.add_argument("--debug-topk", type=int, default=5, help="How many top labels to print when --debug-scores is set")
    ap.add_argument("--debug-prompts", action="store_true", help="Also print top matching prompts/synonyms (slower)")
    ap.add_argument("--rich-prompts", action="store_true", help="Use multiple prompt templates per synonym (slightly slower)")
    ap.add_argument("--ignore-weights", action="store_true", help="Ignore label weights (treat all as 1.0) to reduce bias")
    ap.add_argument("--skip-existing", action="store_true", help="Skip files if destination already exists (useful for incremental runs)")
    ap.add_argument("--no-overwrite", action="store_true", help="Error if destination exists instead of auto-renaming (safety guard)")
    ap.add_argument("--date-folders", action="store_true", help="Organize photos into YYYY/MM subfolders based on EXIF or file date")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve() if args.labels else None
    assert src.exists() and src.is_dir(), f"Source folder not found: {src}"
    
    # Load labels (built-in or custom)
    if labels_path:
        print(f"Using custom labels from: {labels_path}")
    else:
        print("Using built-in labels")
    
    labels_cfg = load_labels(labels_path, use_builtin=True)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    pretrained = "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    text_features, owners, prompts = build_text_tokens(model, tokenizer, labels_cfg, device, args.rich_prompts, args.ignore_weights)

    seen_hashes = set()
    duplicate_dir = dst / "_duplicates"

    files = [p for p in src.rglob("*") if p.is_file() and is_image_file(p)]
    if not files:
        print("No images found.")
        return

    action = "COPY" if args.copy else "MOVE"
    print(f"Device: {device} | Label count: {len(labels_cfg)} | Prompts: {len(prompts)} | Agg: {args.agg}")
    print(f"Processing {len(files)} files... ({action}, topk={args.topk}, threshold={args.threshold}, dry_run={args.dry_run})")

    for f in tqdm(files):
        # Dedupe first
        if args.dedupe:
            ah = compute_phash(f)
            if ah is not None:
                if ah in seen_hashes:
                    target = duplicate_dir / f.name
                    if args.dry_run:
                        print(f"[DUP] {f} -> {target}")
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        safe_move(f, target, dry_run=False, skip_existing=args.skip_existing, no_overwrite=args.no_overwrite)
                    continue
                seen_hashes.add(ah)

        # Tagging
        try:
            image_t = load_image(f, device, preprocess)
            scores, probs = score_labels(model, image_t, text_features, owners, agg=args.agg)
        except Exception as e:
            print(f"[WARN] Failed model inference on {f}: {e}")
            scores = {}
            probs = []

        # Debug prints
        if args.debug_scores and scores:
            sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            print(f"[DBG] {f.name} top{args.debug_topk}:", [(k, round(v, 4)) for k, v in sorted_scores[:args.debug_topk]])
            if args.debug_prompts and len(probs) > 0:
                # show top prompt hits with (label/synonym)
                top_pairs = sorted([(float(probs[i]), owners[i][0], owners[i][2], prompts[i]) for i in range(len(prompts))], reverse=True)[:args.debug_topk]
                print("       prompts:", [(round(p, 4), lab, syn) for p, lab, syn, _ in top_pairs])

        # Decide destination label(s)
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen: List[str] = []
        if sorted_scores:
            (top_label, top_score) = sorted_scores[0]
            second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

            # decision policies
            accept = False
            if args.decision == "threshold":
                accept = top_score >= args.threshold
            elif args.decision == "margin":
                accept = (top_score - second_score) >= args.margin
            elif args.decision == "ratio":
                denom = max(second_score, 1e-9)
                accept = (top_score / denom) >= args.ratio
            elif args.decision == "always-top1":
                accept = True

            if accept:
                chosen = [top_label] if args.topk == 1 else [l for l, _ in sorted_scores[: args.topk]]
            else:
                # soft fallback: if fairly confident still take top-1
                if top_score >= (args.threshold * 0.6):
                    chosen = [top_label]

        if not chosen:
            chosen = ["_unsorted"]

        # Date subfolder (if enabled)
        if args.date_folders:
            year, month = get_exif_datetime(f)
            ydir = f"{year:04d}/{month:02d}"
        else:
            ydir = ""

        # Move/copy
        for label in chosen:
            if ydir:
                out_dir = dst / label / ydir
            else:
                out_dir = dst / label
            target = out_dir / f.name
            try:
                if args.copy and len(chosen) > 1:
                    if args.dry_run:
                        print(f"[COPY] {f} -> {target}")
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        safe_copy(f, target, dry_run=False, skip_existing=args.skip_existing, no_overwrite=args.no_overwrite)
                else:
                    if args.copy:
                        if args.dry_run:
                            print(f"[COPY] {f} -> {target}")
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            safe_copy(f, target, dry_run=False, skip_existing=args.skip_existing, no_overwrite=args.no_overwrite)
                    else:
                        if args.dry_run:
                            print(f"[MOVE] {f} -> {target}")
                        else:
                            safe_move(f, target, dry_run=False, skip_existing=args.skip_existing, no_overwrite=args.no_overwrite)
                    break
            except FileExistsError as e:
                print(f"[ERROR] {e}")
                if args.no_overwrite:
                    print("Stopping due to --no-overwrite flag.")
                    return
                break


if __name__ == "__main__":
    main()
