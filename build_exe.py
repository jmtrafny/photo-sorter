"""
PyInstaller build script for Photo Organizer.
Run this to create a standalone executable.

Usage: python build_exe.py
"""

import PyInstaller.__main__
import sys
from pathlib import Path

def build_exe():
    """Build single user-friendly executable."""
    print("Building Photo Organizer...")
    
    PyInstaller.__main__.run([
        '--name=PhotoOrganizer',
        '--onefile', 
        # Removed --windowed to allow subprocess to work properly
        '--add-data=default_labels.py;.',
        '--add-data=app_streamlit.py;.',
        '--add-data=photo_sorter.py;.',
        '--hidden-import=streamlit',
        '--hidden-import=open_clip',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=PIL',
        '--hidden-import=imagehash',
        '--hidden-import=websockets',
        '--collect-all=streamlit',
        '--collect-all=open_clip',
        '--collect-all=torch',
        '--optimize=2',
        'launch_ui.py'
    ])
    
    print("[OK] PhotoOrganizer.exe built successfully!")

def main():
    print("Building Photo Organizer for end users...")
    print("Creating simple double-click application...")
    print()
    
    build_exe()
    
    print()
    print("Build complete!")
    print("Your app: dist/PhotoOrganizer.exe")
    print()
    print("[NOTE] First run requires internet for model download (~150MB)")
    print("Perfect for non-technical users - just double-click and go!")

if __name__ == "__main__":
    try:
        import PyInstaller
    except ImportError:
        print("[ERROR] PyInstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)
    
    main()