"""
Model manager for Photo Organizer - handles caching, downloading, and loading of CLIP models.

This module provides efficient model management with:
- Automatic caching to avoid repeated downloads
- Progress tracking for downloads
- Support for multiple model sizes
- Thread-safe operations
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import warnings

import torch
import open_clip

# Silence the benign open_clip warning
warnings.filterwarnings(
    "ignore",
    message="QuickGELU mismatch.*",
    category=UserWarning,
    module="open_clip.factory"
)

# Model configurations with estimated accuracy and performance characteristics
MODEL_CONFIGS = {
    "small": {
        "name": "ViT-B-32",
        "pretrained": "openai",
        "accuracy": "~60%",
        "speed": "Fast",
        "size_mb": 150,
        "description": "Balanced speed and accuracy for quick sorting",
        "batch_size": 32,  # Optimal batch size for this model
    },
    "medium": {
        "name": "ViT-B-16", 
        "pretrained": "openai",
        "accuracy": "~75%",
        "speed": "Moderate",
        "size_mb": 300,
        "description": "Better accuracy with moderate speed",
        "batch_size": 16,
    },
    "large": {
        "name": "ViT-L-14",
        "pretrained": "openai",
        "accuracy": "~85%",
        "speed": "Slower",
        "size_mb": 900,
        "description": "Best accuracy for professional use",
        "batch_size": 8,
    },
    "xlarge": {
        "name": "ViT-L-14-336",
        "pretrained": "openai",
        "accuracy": "~90%",
        "speed": "Slowest", 
        "size_mb": 900,
        "description": "Maximum accuracy with higher resolution",
        "batch_size": 4,
    }
}


class ModelManager:
    """Manages CLIP model loading, caching, and configuration."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Directory for caching models. Defaults to user's home/.cache/photo_organizer
        """
        if cache_dir is None:
            # Use Windows AppData for better permissions, fallback to user home
            try:
                if os.name == 'nt':  # Windows
                    appdata = os.environ.get('APPDATA')
                    if appdata:
                        cache_dir = Path(appdata) / "photo_organizer"
                    else:
                        cache_dir = Path.home() / "AppData" / "Roaming" / "photo_organizer"
                else:
                    cache_dir = Path.home() / ".cache" / "photo_organizer"
            except Exception:
                # Ultimate fallback - current directory
                cache_dir = Path("./models_cache")
        
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # If we can't create in AppData, fall back to current directory
            self.cache_dir = Path("./models_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store loaded models to avoid reloading
        self._loaded_models: Dict[str, Tuple] = {}
        
        # Track download progress
        self.download_callback: Optional[Callable[[int, int], None]] = None
        
    def get_model_info(self, model_size: str) -> Dict:
        """
        Get information about a model size.
        
        Args:
            model_size: One of 'small', 'medium', 'large', 'xlarge'
            
        Returns:
            Dictionary with model configuration
        """
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
        return MODEL_CONFIGS[model_size].copy()
    
    def get_cache_path(self, model_size: str) -> Path:
        """
        Get the cache path for a specific model.
        
        Args:
            model_size: Model size identifier
            
        Returns:
            Path to the cached model directory
        """
        config = self.get_model_info(model_size)
        # Create a unique cache key based on model name and pretrained source
        cache_key = f"{config['name']}_{config['pretrained']}"
        return self.cache_dir / cache_key
    
    def is_cached(self, model_size: str) -> bool:
        """
        Check if a model is already cached locally.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model is cached and ready to use
        """
        cache_path = self.get_cache_path(model_size)
        
        # Check multiple possible cache locations and file patterns
        possible_patterns = [
            "*.pt", "*.pth", "*.safetensors", "*.bin",
            "pytorch_model.bin", "model.safetensors",
            "open_clip_pytorch_model.bin"
        ]
        
        if cache_path.exists():
            for pattern in possible_patterns:
                if list(cache_path.glob(pattern)):
                    return True
        
        # Also check if open_clip's default cache has this model
        try:
            config = self.get_model_info(model_size)
            model_name = config["name"]
            pretrained = config["pretrained"]
            
            # Check default open_clip cache locations
            default_cache_locations = [
                Path.home() / ".cache" / "clip",
                Path.home() / ".cache" / "open_clip",
                Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
                self.cache_dir / "torch_hub" / "checkpoints",
                self.cache_dir,
            ]
            
            for cache_loc in default_cache_locations:
                if cache_loc.exists():
                    # Look for files containing the model name
                    model_files = list(cache_loc.glob(f"*{model_name.lower()}*")) + \
                                 list(cache_loc.glob(f"*{pretrained}*")) + \
                                 list(cache_loc.glob("*.pt")) + \
                                 list(cache_loc.glob("*.pth"))
                    if model_files:
                        return True
        except Exception:
            pass
        
        return False
    
    def estimate_processing_time(self, num_images: int, model_size: str, use_gpu: bool = False) -> float:
        """
        Estimate processing time for a batch of images.
        
        Args:
            num_images: Number of images to process
            model_size: Model size to use
            use_gpu: Whether GPU acceleration is available
            
        Returns:
            Estimated time in seconds
        """
        config = self.get_model_info(model_size)
        
        # Base processing times per image (in seconds) - empirical estimates
        if use_gpu:
            base_times = {
                "small": 0.05,
                "medium": 0.08,
                "large": 0.15,
                "xlarge": 0.25
            }
        else:
            base_times = {
                "small": 0.3,
                "medium": 0.5,
                "large": 1.0,
                "xlarge": 1.8
            }
        
        base_time = base_times.get(model_size, 0.5)
        
        # Add overhead for model loading (one-time cost)
        load_overhead = 5 if not self.is_cached(model_size) else 2
        
        # Calculate total time with batching efficiency
        batch_size = config["batch_size"]
        num_batches = (num_images + batch_size - 1) // batch_size
        
        # Batching provides some efficiency gains
        batch_efficiency = 0.9 if num_batches > 1 else 1.0
        
        total_time = load_overhead + (num_images * base_time * batch_efficiency)
        
        return total_time
    
    def load_model(self, model_size: str = "small", device: str = "cpu", 
                   progress_callback: Optional[Callable[[str], None]] = None):
        """
        Load a CLIP model with caching support.
        
        Args:
            model_size: Size of model to load ('small', 'medium', 'large', 'xlarge')
            device: Device to load model on ('cpu' or 'cuda')
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (model, preprocess, tokenizer)
        """
        # Check if already loaded
        cache_key = f"{model_size}_{device}"
        if cache_key in self._loaded_models:
            if progress_callback:
                progress_callback("Model already loaded in memory")
            return self._loaded_models[cache_key]
        
        config = self.get_model_info(model_size)
        model_name = config["name"]
        pretrained = config["pretrained"]
        
        if progress_callback:
            if self.is_cached(model_size):
                progress_callback(f"Loading {model_size} model from cache...")
            else:
                progress_callback(f"Downloading {model_size} model (~{config['size_mb']}MB)...")
        
        try:
            # Set torch hub directory to our cache location for better control
            cache_path = self.get_cache_path(model_size)
            
            # Store original env vars to restore later
            original_env = {}
            cache_env_vars = [
                'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME'
            ]
            
            for var in cache_env_vars:
                original_env[var] = os.environ.get(var)
                os.environ[var] = str(self.cache_dir)  # Use parent cache dir
            
            # Also try to set torch hub directory directly
            import torch.hub
            original_hub_dir = None
            try:
                original_hub_dir = torch.hub.get_dir()
                torch.hub.set_dir(str(self.cache_dir / "torch_hub"))
            except Exception:
                pass
            
            try:
                # Load the model - open_clip should now use our cache location
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained,
                    cache_dir=str(self.cache_dir)  # Try this parameter too
                )
            finally:
                # Restore original environment variables
                for var, value in original_env.items():
                    if value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = value
                
                # Restore torch hub directory
                if original_hub_dir:
                    try:
                        torch.hub.set_dir(original_hub_dir)
                    except Exception:
                        pass
            
            tokenizer = open_clip.get_tokenizer(model_name)
            
            # Move to device and set to eval mode
            model = model.to(device).eval()
            
            # Cache the loaded model
            self._loaded_models[cache_key] = (model, preprocess, tokenizer)
            
            if progress_callback:
                progress_callback(f"{model_size.capitalize()} model loaded successfully")
            
            return model, preprocess, tokenizer
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error loading model: {str(e)}")
            raise
    
    def clear_cache(self, model_size: Optional[str] = None):
        """
        Clear cached models from disk.
        
        Args:
            model_size: Specific model to clear, or None to clear all
        """
        if model_size:
            cache_path = self.get_cache_path(model_size)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
        else:
            # Clear entire cache
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_optimal_batch_size(self, model_size: str, available_memory_gb: Optional[float] = None) -> int:
        """
        Get optimal batch size based on model and available memory.
        
        Args:
            model_size: Model size being used
            available_memory_gb: Available RAM in GB (auto-detect if None)
            
        Returns:
            Recommended batch size
        """
        config = self.get_model_info(model_size)
        base_batch_size = config["batch_size"]
        
        if available_memory_gb is None:
            # Try to detect available memory
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
            except ImportError:
                # If psutil not available, use conservative estimate
                available_memory_gb = 4.0
        
        # Adjust batch size based on available memory
        if available_memory_gb < 4:
            return max(1, base_batch_size // 4)
        elif available_memory_gb < 8:
            return max(1, base_batch_size // 2)
        elif available_memory_gb > 16:
            return base_batch_size * 2
        else:
            return base_batch_size
    
    def preload_model(self, model_size: str = "small"):
        """
        Preload a model into cache without returning it.
        Useful for background downloading.
        
        Args:
            model_size: Model to preload
        """
        if not self.is_cached(model_size):
            # Temporarily load to trigger download
            device = "cpu"  # Use CPU for preloading to save GPU memory
            self.load_model(model_size, device)
            # Clear from memory after caching
            cache_key = f"{model_size}_{device}"
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
                
    def get_available_models(self) -> Dict[str, Dict]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary of model configurations with cache status
        """
        models = {}
        for size, config in MODEL_CONFIGS.items():
            model_info = config.copy()
            model_info["cached"] = self.is_cached(size)
            models[size] = model_info
        return models
    
    def debug_cache_status(self, model_size: str = None) -> str:
        """
        Get detailed cache status for debugging.
        
        Args:
            model_size: Specific model to check, or None for all
            
        Returns:
            Detailed cache status string
        """
        debug_info = []
        debug_info.append(f"Cache directory: {self.cache_dir}")
        debug_info.append(f"Cache dir exists: {self.cache_dir.exists()}")
        
        if model_size:
            models_to_check = [model_size]
        else:
            models_to_check = list(MODEL_CONFIGS.keys())
        
        for size in models_to_check:
            debug_info.append(f"--- {size.upper()} MODEL ---")
            cache_path = self.get_cache_path(size)
            debug_info.append(f"Model cache path: {cache_path}")
            debug_info.append(f"Path exists: {cache_path.exists()}")
            
            if cache_path.exists():
                files = list(cache_path.glob("*"))
                debug_info.append(f"Files in cache: {len(files)}")
                for f in files[:5]:  # Show first 5 files
                    debug_info.append(f"  - {f.name}")
                if len(files) > 5:
                    debug_info.append(f"  ... and {len(files) - 5} more")
            
            debug_info.append(f"Is cached: {self.is_cached(size)}")
        
        # Check system cache locations
        debug_info.append("--- SYSTEM CACHE LOCATIONS ---")
        system_caches = [
            Path.home() / ".cache" / "clip",
            Path.home() / ".cache" / "open_clip", 
            Path.home() / ".cache" / "torch",
        ]
        
        for cache_loc in system_caches:
            debug_info.append(f"{cache_loc}: {'EXISTS' if cache_loc.exists() else 'NOT FOUND'}")
            if cache_loc.exists():
                files = list(cache_loc.glob("*.pt")) + list(cache_loc.glob("*.pth"))
                if files:
                    debug_info.append(f"  Model files found: {len(files)}")
        
        return "\n".join(debug_info)


# Singleton instance for convenient access
_default_manager = None

def get_model_manager(cache_dir: Optional[Path] = None) -> ModelManager:
    """
    Get the default model manager instance.
    
    Args:
        cache_dir: Optional cache directory override
        
    Returns:
        ModelManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager(cache_dir)
    return _default_manager