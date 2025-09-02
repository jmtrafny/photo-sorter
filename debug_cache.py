#!/usr/bin/env python3
"""
Debug script to check model caching status and paths.
Use this to troubleshoot caching issues.
"""

from model_manager import get_model_manager

def main():
    print("=== PHOTO ORGANIZER CACHE DEBUG ===\n")
    
    # Initialize model manager
    model_manager = get_model_manager()
    
    # Print debug information
    debug_info = model_manager.debug_cache_status()
    print(debug_info)
    
    print("\n=== TESTING CACHE DETECTION ===")
    
    for model_size in ["small", "medium", "large"]:
        is_cached = model_manager.is_cached(model_size)
        print(f"{model_size.upper()} model cached: {'YES' if is_cached else 'NO'}")
    
    print("\n=== CACHE PATHS ===")
    for model_size in ["small", "medium", "large"]:
        cache_path = model_manager.get_cache_path(model_size)
        print(f"{model_size.upper()}: {cache_path}")

if __name__ == "__main__":
    main()