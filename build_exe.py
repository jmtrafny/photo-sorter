"""
PyInstaller build script for Photo Organizer with Tkinter UI.
This creates a truly standalone executable that doesn't require Python.

Usage: python build_exe.py
"""

import PyInstaller.__main__
import sys
import shutil
from pathlib import Path


def clean_build_dirs():
    """Clean up old build artifacts."""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Cleaning {dir_name}/...")
            shutil.rmtree(dir_path, ignore_errors=True)


def build_exe():
    """Build standalone executable with Tkinter UI."""
    print("=" * 60)
    print("Building Photo Organizer...")
    print("=" * 60)
    
    # Clean old builds
    clean_build_dirs()
    
    # PyInstaller arguments
    args = [
        'app_tkinter.py',  # Entry point is the Tkinter app
        '--name=PhotoOrganizer',
        '--onefile',  # Single executable file
        '--windowed',  # No console window (GUI app)
        '--icon=icon.ico' if Path('icon.ico').exists() else '--icon=NONE',
        
        # Add data files
        '--add-data=default_labels.py;.',
        '--add-data=photo_sorter.py;.',  # Include CLI tool for subprocess
        
        # Hidden imports for dependencies
        '--hidden-import=PIL',
        '--hidden-import=PIL.Image',
        '--hidden-import=PIL.ImageTk',
        '--hidden-import=open_clip',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=numpy',
        '--hidden-import=imagehash',
        '--hidden-import=tqdm',
        
        # Collect all required packages
        '--collect-all=open_clip',
        '--collect-all=torch',
        '--collect-all=PIL',
        
        # Optimization
        '--optimize=2',
        
        # Paths
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
        
        # Suppress warnings
        '--noconfirm',
    ]
    
    print("\nRunning PyInstaller with arguments:")
    for arg in args:
        if not arg.startswith('--'):
            print(f"  Input: {arg}")
        else:
            print(f"  {arg}")
    print()
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("\n" + "=" * 60)
    print("[OK] Build completed successfully!")
    print("=" * 60)
    
    # Check output
    exe_path = Path('dist/PhotoOrganizer.exe')
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n[PACKAGE] Executable created: {exe_path}")
        print(f"[INFO] File size: {size_mb:.1f} MB")
        print("\n[FEATURES]")
        print("  - Standalone executable - no Python required")
        print("  - Native Windows GUI with Tkinter")
        print("  - All ML models included")
        print("  - Settings saved to user profile")
        print("\n[NOTES]")
        print("  - First run downloads CLIP model (~150MB)")
        print("  - Model cached for subsequent runs")
        print("  - Perfect for non-technical users!")
    else:
        print("\n[ERROR] Executable not found in dist/ folder")
        return 1
    
    return 0


def main():
    """Main build process."""
    print("Photo Organizer - Build Script")
    print("Creating standalone Windows executable")
    print()
    
    # Check for required files
    required_files = ['app_tkinter.py', 'photo_sorter.py', 'default_labels.py']
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print("[ERROR] Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease ensure all files are present before building.")
        return 1
    
    # Check for PyInstaller
    try:
        import PyInstaller
        print(f"[OK] PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("[ERROR] PyInstaller not found")
        print("Install with: pip install pyinstaller")
        return 1
    
    # Run build
    result = build_exe()
    
    if result == 0:
        print("\n[SUCCESS] Build successful! You can now distribute PhotoOrganizer.exe")
        print("   Users can simply double-click to run - no installation needed!")
    
    return result


if __name__ == "__main__":
    sys.exit(main())