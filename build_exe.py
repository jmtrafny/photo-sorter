"""
PyInstaller build script for Photo Organizer.
Run this to create a standalone executable.

Usage: python build_exe.py
"""

import PyInstaller.__main__
import sys
from pathlib import Path

def build_cli_exe():
    """Build CLI executable."""
    PyInstaller.__main__.run([
        '--name=PhotoOrganizer-CLI',
        '--onefile',
        '--console',
        '--add-data=default_labels.py;.',
        '--hidden-import=open_clip',
        '--hidden-import=torch',
        '--hidden-import=torchvision', 
        '--hidden-import=PIL',
        '--hidden-import=imagehash',
        '--hidden-import=tqdm',
        '--collect-all=open_clip',
        '--collect-all=torch',
        '--optimize=2',
        'photo_sorter.py'
    ])

def build_ui_exe():
    """Build UI executable."""
    PyInstaller.__main__.run([
        '--name=PhotoOrganizer-UI',
        '--onefile', 
        '--windowed',  # No console window
        '--add-data=default_labels.py;.',
        '--add-data=app_streamlit.py;.',
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

def main():
    print("Building Photo Organizer executables...")
    print()
    
    choice = input("Build (1) CLI only, (2) UI only, or (3) both? [3]: ").strip() or "3"
    
    if choice in ["1", "3"]:
        print("Building CLI executable...")
        build_cli_exe()
        print("[OK] CLI executable built: dist/PhotoOrganizer-CLI.exe")
    
    if choice in ["2", "3"]:
        print("Building UI executable...")
        build_ui_exe() 
        print("[OK] UI executable built: dist/PhotoOrganizer-UI.exe")
    
    print()
    print("Build complete!")
    print("Executables are in the 'dist' folder")
    print()
    print("[NOTE] First run requires internet for model download (~150MB)")
    print("Tip: Test executables before distributing")

if __name__ == "__main__":
    try:
        import PyInstaller
    except ImportError:
        print("[ERROR] PyInstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)
    
    main()