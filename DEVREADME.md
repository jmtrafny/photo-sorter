# Photo Organizer - Developer Documentation

## Overview

Photo Organizer is an AI-powered tool for automatically sorting photos using zero-shot image classification. It uses OpenCLIP (OpenAI's CLIP model) to analyze image content locally and organize photos into categories without requiring any training data.

### Key Features

* **Zero-shot classification**: No training required, works on any photo collection
* **Local processing**: No cloud APIs, all computation happens on your machine
* **Multiple interfaces**: Both CLI and web UI available
* **Flexible organization**: Category-based or date-based folder structures
* **Duplicate detection**: Perceptual hashing to identify near-duplicates
* **Built-in labels**: Works out-of-the-box with sensible defaults

## Architecture

### Core Components

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Input    │     │  CLIP Model     │     │   File System   │
│                 │     │                 │     │                 │
│ CLI/Streamlit   │───▶│ Image Encoder   │───▶│  Sorted Photos   │
│ UI              │     │ Text Encoder    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │
         │               ┌──────────────────┐
         └─────────────▶│ Label Config     │
                         │ (built-in/custom)│
                         └──────────────────┘
```

### File Structure

```

├── photo_sorter.py      # CLI implementation
├── app_streamlit.py     # Web UI implementation
├── default_labels.py    # Built-in label configurations
├── undo.py             # Undo functionality
├── README.md           # User documentation
├── DEVREADME.md        # This file
└── labels.json         # Example custom labels (optional)
```

## Core Technologies

### 1\. OpenCLIP \(Computer Vision\)

**What it is**: An open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) model.

**How it works**:

* Encodes images and text into the same embedding space
* Enables zero-shot classification by comparing image embeddings to text embeddings
* Uses cosine similarity to measure relevance between images and text prompts

**Model used**: `ViT-B/32` (Vision Transformer with 32×32 patches)

* Balance between accuracy and speed
* \~150MB model size
* Good performance on diverse image types

**Key files**: `photo_sorter.py:build_text_tokens()`, `score_labels()`

**Resources**:

* [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
* [CLIP Paper](https://arxiv.org/abs/2103.00020)
* [Hugging Face OpenCLIP](https://huggingface.co/docs/transformers/model_doc/clip)

### 2\. Streamlit \(Web UI\)

**What it is**: Python framework for building data apps with minimal frontend code.

**Why chosen**:

* Rapid prototyping for ML applications
* Built-in caching for expensive operations (model loading)
* Real-time parameter tuning with widgets
* Easy deployment

**Key features used**:

* `@st.cache_resource`: Cache model loading
* `@st.cache_data`: Cache file scanning
* Sidebar widgets for parameters
* Real-time subprocess output streaming

**Key files**: `app_streamlit.py`

**Resources**:

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Streamlit Gallery](https://streamlit.io/gallery)

### 3\. PIL/Pillow \(Image Processing\)

**What it is**: Python Imaging Library for image manipulation.

**Usage**:

* Loading and preprocessing images for CLIP
* EXIF metadata extraction for date organization
* Format conversion (ensure RGB)

**Key files**: `photo_sorter.py:load_image()`, `get_exif_datetime()`

**Resources**:

* [Pillow Documentation](https://pillow.readthedocs.io/)
* [EXIF Tags Reference](https://exiv2.org/tags.html)

### 4\. ImageHash \(Duplicate Detection\)

**What it is**: Perceptual hashing library for finding similar images.

**How it works**:

* Creates compact hash representing image content
* Similar images have similar hashes (even with different sizes/compression)
* Uses average hash algorithm for speed

**Key files**: `photo_sorter.py:compute_phash()`

**Resources**:

* [ImageHash GitHub](https://github.com/JohannesBuchner/imagehash)
* [Perceptual Hashing Explained](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)

## Code Organization

### photo\_sorter.py (CLI)

**Main functions**:

* `main()`: Entry point, argument parsing
* `load_labels()`: Load and validate label configurations
* `build_text_tokens()`: Encode all text prompts with CLIP
* `score_labels()`: Compare image to text embeddings
* `get_exif_datetime()`: Extract date from image metadata
* `safe_move()`/`safe_copy()`: File operations with collision handling

**Processing flow**:

1. Load CLIP model and text embeddings
2. Scan for image files recursively
3. For each image:
    * Check for duplicates (if enabled)
    * Classify with CLIP
    * Apply decision policy
    * Move/copy to appropriate folder

### app\_streamlit.py (Web UI)

**Main functions**:

* `get_model_and_text_features()`: Cached model/embeddings loading
* `pick_directory()`: Native folder picker integration
* `build_cli_command()`: Translate UI settings to CLI args
* `run_cli()`: Execute CLI subprocess with real-time output

**UI Architecture**:

* Sidebar: Configuration and controls
* Main area: Preview and results
* WebSocket: Browser connection monitoring for auto-shutdown

**Caching strategy**:

* Model loading: Cached by label config and parameters
* File scanning: Cached by directory path
* Text encoding: Part of model cache

### default\_labels.py

**Purpose**: Provides built-in label configurations eliminating need for external files.

**Categories included**:

* people, pets, landscape, food, documents
* screenshots, cars, flowers, kids, events

**Design principles**:

* Balanced synonym lists (avoid bias)
* Cover common photo types
* Weight adjustment for over/under-active categories

## Development Patterns

### Error Handling

**Philosophy**: Graceful degradation with user feedback

* Invalid files are skipped with warnings
* Malformed configs fall back to built-in labels
* Processing errors don't stop the entire operation
* Clear error messages guide user actions

**Examples**:

``` python
# Graceful fallback for labels
try:
    data = json.load(f)
    if not validate_labels(data):
        print("[WARNING] Invalid labels, using built-in")
        data = None
except json.JSONDecodeError:
    print("[WARNING] Parse error, using built-in")
    data = None
```

### Validation

**Input validation**:

* File paths: Check existence and permissions
* JSON configs: Validate structure and types
* Parameters: Range checking on thresholds/ratios

**Model validation**:

* Device availability (CUDA vs CPU)
* Model loading success
* Text encoding success

### Performance Optimization

**Text encoding**: Done once upfront, not per-image
**Caching**: Expensive operations cached in Streamlit
**Batch processing**: Process all images in single loop
**Early exit**: Skip expensive classification for duplicates

## Decision Logic

The classification system uses configurable decision policies:

### Policies

1. **Threshold**: Accept if `top_score >= threshold`
2. **Margin**: Accept if `(top1 - top2) >= margin`
3. **Ratio**: Accept if `(top1 / top2) >= ratio`
4. **Always-top1**: Always accept highest score

### Score Aggregation

Multiple synonyms per label require aggregation:

* **Max**: Take highest score (prevents synonym-count bias)
* **Mean**: Average all scores
* **Sum**: Add all scores (can bias toward labels with many synonyms)

### Normalization

Scores are normalized across all labels to sum to 1.0, making thresholds comparable across different label sets.

## Configuration

### Label Configuration Format

``` json
{
  "category_name": {
    "prompt": "base prompt text",
    "synonyms": ["synonym1", "synonym2", ...],
    "weight": 1.0
  }
}
```

### Recommended Practices

**Synonym selection**:

* Include both formal and casual terms
* Cover different visual contexts
* Avoid overlap between categories
* Test with your specific image collection

**Weight tuning**:

* Start with 1.0 for all categories
* Increase weights for under-represented categories
* Decrease weights for over-active categories
* Use `--debug-scores` to analyze performance

## Packaging for Distribution

### EXE Creation

The application is designed to be packaged into standalone executables for easy distribution.

**Benefits of current design**:

* Built-in labels eliminate external file dependencies
* Graceful fallback for missing dependencies
* Clear error messages for troubleshooting
* Unicode-safe for Windows console

### Step-by-Step Packaging Process

#### Prerequisites
```bash
# Install PyInstaller
pip install pyinstaller

# Ensure all dependencies are installed
pip install -r requirements.txt
```

#### Option 1: Automated Build (Recommended)
```bash
# Run the build script
python build_exe.py

# Choose option:
# 1 = CLI only
# 2 = UI only  
# 3 = Both (recommended)
```

#### Option 2: Manual PyInstaller Commands
```bash
# For CLI executable
pyinstaller --name=PhotoOrganizer-CLI --onefile --console --add-data=default_labels.py;. --hidden-import=open_clip --hidden-import=torch --hidden-import=PIL --hidden-import=imagehash --collect-all=open_clip --collect-all=torch photo_sorter.py

# For UI executable  
pyinstaller --name=PhotoOrganizer-UI --onefile --windowed --add-data=default_labels.py;. --add-data=app_streamlit.py;. --hidden-import=streamlit --hidden-import=open_clip --hidden-import=torch --hidden-import=PIL --hidden-import=websockets --collect-all=streamlit --collect-all=open_clip --collect-all=torch launch_ui.py
```

### Output Structure
After building, you'll have:
```
dist/
├── PhotoOrganizer-CLI.exe    # Command line version (~500MB)
├── PhotoOrganizer-UI.exe     # Web interface version (~500MB)
└── [build artifacts]
```

### Creating Distribution Package for GitHub
```bash
# 1. Create distribution folder
mkdir PhotoOrganizer-Release

# 2. Copy executables
copy dist\PhotoOrganizer-CLI.exe PhotoOrganizer-Release\
copy dist\PhotoOrganizer-UI.exe PhotoOrganizer-Release\

# 3. Copy documentation
copy README.md PhotoOrganizer-Release\
copy labels.json PhotoOrganizer-Release\example-labels.json

# 4. Create usage guide
echo "Quick Start:" > PhotoOrganizer-Release\USAGE.txt
echo "1. Double-click PhotoOrganizer-UI.exe for web interface" >> PhotoOrganizer-Release\USAGE.txt
echo "2. Or use PhotoOrganizer-CLI.exe for command line" >> PhotoOrganizer-Release\USAGE.txt
echo "3. First run downloads model (~150MB) - requires internet" >> PhotoOrganizer-Release\USAGE.txt
```

### Key Considerations
* **Size**: ~500MB per executable due to PyTorch/CLIP
* **First Run**: Requires internet for model download (~150MB)  
* **Memory**: 4GB+ RAM recommended for large collections
* **GPU**: CUDA support adds significant size if included
* **Compatibility**: Windows 10+ (can build for other platforms)

### Dependencies

**Core requirements**:

```
torch>=1.13.0
open-clip-torch>=2.20.0
streamlit>=1.28.0
pillow>=9.0.0
imagehash>=4.3.0
tqdm>=4.64.0
websockets>=12.0
```

**Optional**:

* tkinter (usually bundled with Python)
* CUDA libraries (for GPU acceleration)

## Testing Strategy

### Manual Testing Checklist

**CLI testing**:

* [ ] Dry run with sample images
* [ ] Different decision policies
* [ ] Custom vs built-in labels
* [ ] Error conditions (missing files, invalid JSON)

**UI testing**:

* [ ] Folder picker functionality
* [ ] Preview accuracy
* [ ] Parameter changes reflected in CLI command
* [ ] Auto-shutdown behavior

**Cross-platform**:

* [ ] Windows, macOS, Linux compatibility
* [ ] Path handling (backslashes vs forward slashes)
* [ ] Permission issues

### Performance Testing

**Metrics to monitor**:

* Images processed per second
* Memory usage during processing
* Model loading time
* UI responsiveness during processing

**Test datasets**:

* Small set (\~50 images): Quick iteration
* Medium set (\~500 images): Real usage simulation
* Large set (\~5000+ images): Stress testing

## Deployment Considerations

### System Requirements

**Minimum**:

* 4GB RAM (8GB recommended)
* 2GB disk space
* CPU: Any modern processor

**Optimal**:

* 16GB+ RAM for large collections
* NVIDIA GPU with CUDA support
* SSD for faster image loading

### Security Considerations

**File access**: Application needs read access to source images and write access to destination.

**Network**: Streamlit UI binds to localhost by default (secure).

**Privacy**: All processing is local, no data sent to external services.

## Troubleshooting Guide

### Common Issues

**"No module named open\_clip"**

* Solution: Install requirements.txt dependencies

**"CUDA out of memory"**

* Solution: Process smaller batches or use CPU mode
* Code: Add `--device cpu` flag (if implemented)

**"Permission denied"**

* Solution: Check file/folder permissions
* Common on: Network drives, restricted folders

**Images not being classified correctly**

* Solution: Review and adjust labels
* Debug with: `--debug-scores --debug-prompts` flags

### Debugging Tools

**CLI flags for debugging**:

* `--debug-scores`: Show top classification scores
* `--debug-prompts`: Show which prompts matched
* `--dry-run`: Preview actions without moving files

**Log analysis**:

* Look for [WARNING] messages about failed processing
* Check classification scores for confidence levels
* Review actual vs expected categorization

## Extension Points

### Adding New Categories

1. Edit `default_labels.py` or create custom JSON
2. Test with representative images
3. Adjust weights based on performance
4. Consider synonym overlap with existing categories

### Custom Decision Policies

Extend `decide_label()` function in both CLI and UI:

``` python
elif policy == "custom":
    # Implement your logic here
    accept = custom_decision_logic(scores)
```

### Additional File Formats

Add to `SUPPORTED_EXTS` in both files:

``` python
SUPPORTED_EXTS.add(".heic")  # iPhone photos
```

Ensure PIL/Pillow supports the format.

### Alternative Models

Replace CLIP model in `get_model_and_text_features()`:

* Different CLIP variants (ViT-L/14, RN50, etc.)
* Other vision-language models
* Custom fine-tuned models

## Contributing

### Code Style

* Follow existing patterns and naming conventions
* Add docstrings for all functions
* Include type hints where helpful
* Comment complex logic thoroughly

### Testing

* Test on multiple platforms
* Validate with different image types
* Check edge cases (empty folders, corrupted files)
* Performance test with large collections

### Documentation

* Update README.md for user-facing changes
* Update DEVREADME.md for technical changes
* Include examples for new features
* Document breaking changes clearly

## Resources & References

### Computer Vision & AI

* [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020)
* [Vision Transformers Explained](https://arxiv.org/abs/2010.11929)
* [Zero-shot Learning Survey](https://arxiv.org/abs/1707.00600)

### Python Libraries

* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Pillow Handbook](https://pillow.readthedocs.io/en/stable/handbook/)

### Image Processing

* [Digital Image Processing (Gonzalez & Woods)](https://www.imageprocessingplace.com/)
* [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)

### Development Tools

* [PyInstaller Documentation](https://pyinstaller.readthedocs.io/)
* [Python Packaging Guide](https://packaging.python.org/)
* [Git Workflow Best Practices](https://www.atlassian.com/git/workflows)

- - -

**Last Updated**: 2025-08-30
**Maintainer**: James M. Trafny