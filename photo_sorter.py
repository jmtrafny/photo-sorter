#!/usr/bin/env python3
r"""
photo_sorter.py - CLI tool for AI-powered photo organization

This module provides the command-line interface for sorting photos using OpenCLIP's
zero-shot image classification. It analyzes images locally (no cloud APIs) and
organizes them into categories based on visual content.

Key Features:
- Zero-shot classification using CLIP (no training required)
- Duplicate detection via perceptual hashing
- EXIF-based date organization
- Multiple routing strategies (threshold, margin, ratio)
- Built-in or custom label configurations

Usage Examples:
  # Dry run with built-in labels
  python photo_sorter.py --src "C:\Photos\Unsorted" --dst "C:\Photos\Sorted" --dry-run
  
  # Full run with custom labels and date folders
  python photo_sorter.py --src /photos --dst /sorted --labels my_labels.json --date-folders
  
  # Copy with deduplication
  python photo_sorter.py --src /path/in --dst /path/out --copy --dedupe
"""

import argparse
import json
import shutil
import warnings
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from PIL import Image, ExifTags
import imagehash
import torch
import open_clip
from tqdm import tqdm

from default_labels import DEFAULT_LABELS, get_labels_for_tier
from model_manager import ModelManager, get_model_manager

# Silence the benign open_clip warning about QuickGELU
# This warning occurs because OpenAI's CLIP was trained with QuickGELU activation
# but the config expects regular GELU. The model works fine despite this mismatch.
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
    """Validate labels configuration structure to prevent runtime errors.
    
    Checks that the labels data has the correct JSON structure with proper
    types for all fields. This prevents crashes from malformed user configs.
    
    Args:
        data: Dictionary loaded from labels JSON file
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        If valid, error_message is empty string
    """
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


def load_labels(labels_path: Path | None = None, use_builtin: bool = True, label_tier: str = "small") -> Dict[str, Dict]:
    """Load labels configuration with validation and fallback to built-in defaults.
    
    This function handles the loading of label configurations for image classification.
    It supports both custom user-provided labels and built-in defaults, with automatic
    fallback if custom labels are invalid.
    
    Label Structure:
        {
            "category_name": {
                "prompt": "base prompt text",
                "synonyms": ["synonym1", "synonym2", ...],
                "weight": 1.0  # Optional weight multiplier
            }
        }
    
    Args:
        labels_path: Path to custom labels.json file (optional)
        use_builtin: If True and labels_path is None or invalid, use built-in labels
    
    Returns:
        Dictionary of label configurations with normalized structure
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
            # Use tiered labels based on model size
            data = get_labels_for_tier(label_tier)
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
    """Generate prompt variations for a synonym to improve CLIP's understanding.
    
    CLIP performs better with natural language prompts rather than single words.
    Rich prompts use multiple templates to capture different phrasings, improving
    accuracy at the cost of slightly more computation.
    
    Args:
        syn: The synonym/keyword to create prompts for
        rich: If True, generate multiple prompt templates; if False, use single template
        
    Yields:
        String prompts formatted with the synonym
    """
    if not rich:
        yield f"a photo of {syn}"
        return
    # These templates help CLIP understand context without introducing too much bias
    # Based on OpenAI's CLIP paper and empirical research on prompt effectiveness
    templates = [
        "{}",                      # Direct keyword (works well for objects)
        "a photo of {}",           # Most effective general template
        "a picture of {}",         # Alternative to "image" - more natural language
        "a close-up of {}",        # Shorter, more natural than "close-up photo"
        "{} in a photo",           # Different grammatical structure
        "this is {}",              # Natural identification phrase
    ]
    for t in templates:
        yield t.format(syn)


def build_text_tokens(model, tokenizer, labels_cfg: Dict[str, Dict], device: str, rich_prompts: bool, ignore_weights: bool):
    """Create text embeddings for all label synonyms using CLIP's text encoder.
    
    This function prepares all the text prompts that images will be compared against.
    Each label can have multiple synonyms, and each synonym can generate multiple
    prompt variations. All are encoded into the CLIP embedding space.
    
    The embeddings are normalized to unit vectors for cosine similarity comparison.
    
    Args:
        model: The CLIP model with encode_text capability
        tokenizer: CLIP's tokenizer for converting text to tokens
        labels_cfg: Dictionary of label configurations from load_labels()
        device: 'cuda' or 'cpu' for computation
        rich_prompts: Whether to use multiple prompt templates per synonym
        ignore_weights: If True, treat all labels with weight=1.0
        
    Returns:
        text_features: Tensor of shape (N_prompts, embedding_dim) with normalized embeddings
        owners: List of (label_name, weight, synonym) tuples mapping prompts to labels
        texts: List of original prompt strings (useful for debugging)
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
    """Load and preprocess an image for CLIP model input.
    
    Args:
        path: Path to the image file
        device: Target device ('cuda' or 'cpu')
        preprocess: CLIP's preprocessing transform (resize, normalize, etc.)
        
    Returns:
        Tensor of shape (1, 3, 224, 224) ready for CLIP encoding
    """
    img = Image.open(path).convert("RGB")  # Ensure RGB format
    img_t = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension
    return img_t


def load_images_batch(paths: List[Path], device: str, preprocess, max_workers: int = 4):
    """Load and preprocess multiple images in parallel.
    
    Args:
        paths: List of paths to image files
        device: Target device ('cuda' or 'cpu')
        preprocess: CLIP's preprocessing transform
        max_workers: Number of parallel workers for loading
        
    Returns:
        Tensor of shape (N, 3, 224, 224) ready for CLIP encoding
    """
    def load_single(path):
        try:
            img = Image.open(path).convert("RGB")
            return preprocess(img), path
        except Exception:
            return None, path
    
    # Load images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single, paths))
    
    # Filter out failed loads and stack successful ones
    valid_tensors = []
    valid_paths = []
    for tensor, path in results:
        if tensor is not None:
            valid_tensors.append(tensor)
            valid_paths.append(path)
    
    if valid_tensors:
        # Stack into batch tensor
        batch = torch.stack(valid_tensors).to(device)
        return batch, valid_paths
    else:
        return None, []


def score_labels(model, image_t, text_features, owners, *, agg: str = "max", temperature: float = 1.0):
    """Compute label scores by comparing image to all text prompts.
    
    This is the core classification logic. The image is encoded and compared
    to all text embeddings using cosine similarity. Scores are aggregated
    per label since each label may have multiple prompts.
    
    The temperature parameter controls the sharpness of the probability
    distribution (lower = sharper/more confident).
    
    Args:
        model: CLIP model with encode_image capability
        image_t: Preprocessed image tensor from load_image()
        text_features: Text embeddings from build_text_tokens()
        owners: Mapping of prompts to labels from build_text_tokens()
        agg: Aggregation strategy for multiple prompts per label
            - 'max': Take highest score (prevents synonym count bias)
            - 'mean': Average all scores (balanced but can dilute strong matches)
            - 'sum': Add all scores (can bias toward labels with many synonyms)
        temperature: Softmax temperature for probability scaling
        
    Returns:
        scores: Dict mapping label names to normalized probabilities (sum to 1.0)
        probs: Raw probability tensor for all prompts (useful for debugging)
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


def score_labels_batch(model, images_batch, text_features, owners, *, agg: str = "max", temperature: float = 1.0):
    """Compute label scores for a batch of images.
    
    Process multiple images at once for better GPU utilization.
    
    Args:
        model: CLIP model with encode_image capability  
        images_batch: Batch tensor of shape (N, 3, 224, 224)
        text_features: Text embeddings from build_text_tokens()
        owners: Mapping of prompts to labels
        agg: Aggregation strategy ('max', 'mean', 'sum')
        temperature: Softmax temperature
        
    Returns:
        List of (scores_dict, probs_tensor) tuples, one per image
    """
    with torch.no_grad():
        # Encode all images at once
        image_features = model.encode_image(images_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities for all images
        logits = (image_features @ text_features.T) / temperature
        all_probs = logits.softmax(dim=-1)
    
    # Process each image's scores
    results = []
    for probs in all_probs:
        # Gather per-label values
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
        
        # Normalize to sum=1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        results.append((scores, probs))
    
    return results


def get_exif_datetime(path: Path) -> Tuple[int, int]:
    """Extract year and month from image EXIF data or file modification time.
    
    Tries to get the original photo date from EXIF metadata (DateTimeOriginal tag).
    Falls back to file modification time if EXIF is unavailable or invalid.
    
    Args:
        path: Path to the image file
        
    Returns:
        Tuple of (year, month) as integers
    """
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        
        # Find the DateTimeOriginal tag number
        date_tag = None
        for k, v in ExifTags.TAGS.items():
            if v == "DateTimeOriginal":
                date_tag = k
                break
        
        if date_tag and date_tag in exif:
            # EXIF date format: "2023:12:25 14:30:00"
            dt_str = exif[date_tag]
            # Convert colons in date part to hyphens for parsing
            parts = dt_str.replace(":", "-", 2)  # Only replace first 2 colons
            dt = datetime.fromisoformat(parts)
            return dt.year, dt.month
    except Exception:
        pass  # Fall back to file modification time
    
    # Fallback: use file modification time
    dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt.year, dt.month


def safe_move(src: Path, dst: Path, dry_run: bool, skip_existing: bool = False, no_overwrite: bool = False):
    """Move a file safely with collision handling.
    
    Provides multiple strategies for handling existing files at the destination:
    - Auto-rename: Append __1, __2, etc. to filename (default)
    - Skip: Leave source file in place if destination exists
    - Error: Raise exception if destination exists
    
    Args:
        src: Source file path
        dst: Destination file path
        dry_run: If True, don't actually move anything
        skip_existing: If True, skip files when destination exists
        no_overwrite: If True, raise error when destination exists
    """
    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        return  # Don't actually move in dry-run mode
    
    # Handle existing destination file
    if dst.exists():
        if skip_existing:
            print(f"[SKIP] {src} -> {dst} (destination exists)")
            return
        elif no_overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
    
    # Auto-rename if file exists (default behavior)
    base, ext = dst.stem, dst.suffix
    candidate = dst
    i = 1
    while candidate.exists():
        candidate = dst.with_name(f"{base}__{i}{ext}")  # Append __1, __2, etc.
        i += 1
    
    shutil.move(str(src), str(candidate))




def safe_copy(src: Path, dst: Path, dry_run: bool, skip_existing: bool = False, no_overwrite: bool = False):
    """Copy a file safely with collision handling (same logic as safe_move but preserves source).
    
    Args:
        src: Source file path
        dst: Destination file path  
        dry_run: If True, don't actually copy anything
        skip_existing: If True, skip files when destination exists
        no_overwrite: If True, raise error when destination exists
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    
    if dst.exists():
        if skip_existing:
            print(f"[SKIP] {src} -> {dst} (destination exists)")
            return
        elif no_overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
    
    # Auto-rename with collision-safe naming
    base, ext = dst.stem, dst.suffix
    candidate = dst
    i = 1
    while candidate.exists():
        candidate = dst.with_name(f"{base}__{i}{ext}")
        i += 1
    
    # shutil.copy2 preserves metadata (timestamps, etc.)
    shutil.copy2(str(src), str(candidate))
def is_image_file(p: Path) -> bool:
    """Check if a file is a supported image format.
    
    Args:
        p: Path to check
        
    Returns:
        True if file extension indicates a supported image format
    """
    # Supported formats for PIL/OpenCLIP processing
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def compute_phash(path: Path):
    """Compute perceptual hash for duplicate detection.
    
    Uses average hash algorithm to create a hash that's similar for visually
    similar images, even if they differ in size, compression, or minor edits.
    
    Args:
        path: Path to image file
        
    Returns:
        ImageHash object or None if image can't be processed
    """
    try:
        with Image.open(path) as im:
            # Average hash is robust to minor changes but fast to compute
            return imagehash.average_hash(im)
    except Exception:
        return None  # Skip unprocessable images


# ----------------------------
# Main
# ----------------------------

def process_single_result(f: Path, scores: Dict, probs, args, dst: Path, duplicate_dir: Path):
    """Process classification results for a single image.
    
    Helper function to handle the decision logic and file operations
    after an image has been classified.
    
    Args:
        f: Path to the image file
        scores: Label scores dictionary
        probs: Raw probability tensor
        args: Command line arguments
        dst: Destination directory
        duplicate_dir: Directory for duplicates
    """
    # Debug prints
    if args.debug_scores and scores:
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        print(f"[DBG] {f.name} top{args.debug_topk}:", [(k, round(v, 4)) for k, v in sorted_scores[:args.debug_topk]])
    
    # Decision Logic - determine which category(ies) to sort into
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chosen: List[str] = []
    
    if sorted_scores:
        (top_label, top_score) = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        # Apply decision policy
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
            # Soft fallback
            if top_score >= (args.threshold * 0.6):
                chosen = [top_label]
    
    if not chosen:
        chosen = ["_unsorted"]
    
    # Determine folder structure
    if args.date_folders:
        year, month = get_exif_datetime(f)
        ydir = f"{year:04d}/{month:02d}"
    else:
        ydir = ""
    
    # Move or copy the file
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
                raise

def main():
    """Main CLI entry point for photo sorting application.
    
    Sets up argument parsing, loads the CLIP model, and processes all images
    in the source directory according to the specified parameters.
    """
    ap = argparse.ArgumentParser(
        description="Auto-sort photos by content using local zero-shot tagging (OpenCLIP).",
        epilog="For more details, see README.md or DEVREADME.md"
    )
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
    ap.add_argument("--model-size", choices=["small", "medium", "large", "xlarge"], default="small", help="Model size: small (fast), medium (balanced), large (accurate), xlarge (max accuracy)")
    ap.add_argument("--label-tier", choices=["small", "medium", "large"], default=None, help="Label tier: small (8 labels), medium (16 labels), large (40+ labels). If not specified, uses tier based on model size.")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size for processing (auto-detect if not specified)")
    ap.add_argument("--parallel-load", type=int, default=4, help="Number of parallel workers for image loading")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve() if args.labels else None
    assert src.exists() and src.is_dir(), f"Source folder not found: {src}"
    
    # Load label configurations (built-in or custom)
    # This determines what categories photos will be sorted into
    if labels_path:
        print(f"Using custom labels from: {labels_path}")
    else:
        print("Using built-in labels")
    
    # Determine label tier (explicit tier overrides model-based tier)
    if args.label_tier:
        label_tier = args.label_tier
    else:
        # Default: map model size to label tier
        label_tier = "small" if args.model_size == "small" else "medium" if args.model_size == "medium" else "large"
    
    labels_cfg = load_labels(labels_path, use_builtin=True, label_tier=label_tier)

    # Initialize model manager and load selected model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = get_model_manager()
    
    # Get model info and estimate processing time
    model_info = model_manager.get_model_info(args.model_size)
    print(f"Model: {args.model_size.upper()} ({model_info['description']})")
    print(f"Expected accuracy: {model_info['accuracy']} | Speed: {model_info['speed']}")
    
    # Count images first for time estimation
    print(f"Scanning for images in: {src}")
    files = [p for p in src.rglob("*") if p.is_file() and is_image_file(p)]
    if not files:
        print("No images found.")
        return
    
    # Estimate and display processing time
    estimated_time = model_manager.estimate_processing_time(
        len(files), args.model_size, use_gpu=(device == "cuda")
    )
    print(f"Estimated processing time: {estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")
    
    # Load the model with progress callback
    def progress_callback(msg):
        print(f"[MODEL] {msg}")
    
    print(f"Loading {args.model_size} model on {device}...")
    try:
        model, preprocess, tokenizer = model_manager.load_model(
            args.model_size, device, progress_callback
        )
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load CLIP model: {e}")
        return
    
    # Determine optimal batch size
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = model_manager.get_optimal_batch_size(args.model_size)
    print(f"Batch size: {batch_size}")

    # Pre-encode all text prompts for efficient comparison
    # This is done once upfront rather than per-image
    print("Encoding text prompts...")
    text_features, owners, prompts = build_text_tokens(model, tokenizer, labels_cfg, device, args.rich_prompts, args.ignore_weights)

    # Setup for duplicate detection if enabled
    seen_hashes = set()  # Store perceptual hashes of processed images
    duplicate_dir = dst / "_duplicates"  # Where duplicates will be moved

    # Set up parallel processing threads
    torch.set_num_threads(min(8, torch.get_num_threads()))

    action = "COPY" if args.copy else "MOVE"
    print(f"Device: {device} | Label count: {len(labels_cfg)} | Prompts: {len(prompts)} | Agg: {args.agg}")
    print(f"Processing {len(files)} files... ({action}, topk={args.topk}, threshold={args.threshold}, dry_run={args.dry_run})")
    
    if len(files) == 0:
        print("No supported image files found. Supported formats: .jpg, .jpeg, .png, .webp, .bmp, .tiff, .gif")
        print(f"Searched in: {src}")
        return
        
    if len(files) > 10000:
        print(f"[WARNING] Large collection detected ({len(files)} files). Consider processing in smaller batches.")
        print("          This may take significant time and memory. Press Ctrl+C to cancel if needed.")

    # Process images in batches for better performance
    process_batch = batch_size > 1 and not args.debug_scores
    
    if process_batch:
        # Batch processing mode
        print(f"Processing {len(files)} images in batches of {batch_size}...")
        for i in tqdm(range(0, len(files), batch_size), desc="Processing batches"):
            batch_files = files[i:i+batch_size]
            
            # Filter duplicates if enabled
            if args.dedupe:
                non_dup_files = []
                for f in batch_files:
                    ah = compute_phash(f)
                    if ah is not None:
                        if ah in seen_hashes:
                            target = duplicate_dir / f.name
                            if args.dry_run:
                                print(f"[DUP] {f} -> {target}")
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                safe_move(f, target, dry_run=False, skip_existing=args.skip_existing, no_overwrite=args.no_overwrite)
                        else:
                            seen_hashes.add(ah)
                            non_dup_files.append(f)
                    else:
                        non_dup_files.append(f)
                batch_files = non_dup_files
            
            if not batch_files:
                continue
            
            # Load batch of images in parallel
            try:
                batch_tensor, valid_paths = load_images_batch(
                    batch_files, device, preprocess, max_workers=args.parallel_load
                )
                
                if batch_tensor is None:
                    continue
                
                # Score all images in batch
                batch_results = score_labels_batch(
                    model, batch_tensor, text_features, owners, agg=args.agg
                )
                
                # Process results for each image
                for (scores, probs), f in zip(batch_results, valid_paths):
                    process_single_result(f, scores, probs, args, dst, duplicate_dir)
                    
            except Exception as e:
                print(f"[WARN] Failed batch processing: {e}")
                # Fall back to individual processing for this batch
                for f in batch_files:
                    try:
                        image_t = load_image(f, device, preprocess)
                        scores, probs = score_labels(model, image_t, text_features, owners, agg=args.agg)
                        process_single_result(f, scores, probs, args, dst, duplicate_dir)
                    except Exception as e2:
                        print(f"[WARN] Failed processing {f}: {e2}")
    else:
        # Original single-image processing mode (for debugging or small batches)
        for f in tqdm(files, desc="Processing images"):
            # Step 1: Check for duplicates first (before expensive classification)
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

            # Step 2: AI Classification - determine what's in the image
            try:
                # Load and preprocess image for CLIP
                image_t = load_image(f, device, preprocess)
                # Compare image to all text prompts and get scores per label
                scores, probs = score_labels(model, image_t, text_features, owners, agg=args.agg)
            except Exception as e:
                # If image processing fails (corrupted, unsupported format, etc.)
                print(f"[WARN] Failed model inference on {f}: {e}")
                scores = {}
                probs = []

            # Process the single result using helper function
            process_single_result(f, scores, probs, args, dst, duplicate_dir)


if __name__ == "__main__":
    main()
