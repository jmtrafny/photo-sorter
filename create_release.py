#!/usr/bin/env python3
"""
Create distribution package for GitHub release.
Builds executables and packages them with documentation.

Usage: python create_release.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    print("Creating Photo Organizer release package...")
    print()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Step 1: Check prerequisites
    try:
        import PyInstaller
        print("[OK] PyInstaller found")
    except ImportError:
        print("[ERROR] PyInstaller not found. Install with: pip install pyinstaller")
        return
    
    # Step 2: Build executable
    print("Building PhotoOrganizer.exe...")
    try:
        result = subprocess.run([sys.executable, "build_exe.py"], text=True, capture_output=True)
        if result.returncode != 0:
            print(f"[ERROR] Build failed: {result.stderr}")
            return
        print("[OK] PhotoOrganizer.exe built successfully")
    except Exception as e:
        print(f"[ERROR] Build failed: {e}")
        return
    
    # Step 3: Create release folder
    release_dir = Path("PhotoOrganizer-Release")
    if release_dir.exists():
        print("Removing existing release folder...")
        shutil.rmtree(release_dir)
    
    release_dir.mkdir()
    print(f"[OK] Created release folder: {release_dir}")
    
    # Step 4: Copy executables
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("[ERROR] dist folder not found. Build may have failed.")
        return
    
    exe_files = list(dist_dir.glob("*.exe"))
    if not exe_files:
        print("[ERROR] No executables found in dist folder")
        return
    
    for exe_file in exe_files:
        shutil.copy2(exe_file, release_dir)
        print(f"[OK] Copied: {exe_file.name}")
    
    # Step 5: Copy documentation
    docs_to_copy = ["README.md"]
    for doc in docs_to_copy:
        if Path(doc).exists():
            shutil.copy2(doc, release_dir)
            print(f"[OK] Copied: {doc}")
    
    # Copy example labels if exists
    if Path("labels.json").exists():
        shutil.copy2("labels.json", release_dir / "example-labels.json")
        print("[OK] Copied: labels.json -> example-labels.json")
    
    # Step 6: Create usage guide
    usage_content = """Photo Organizer - AI-Powered Photo Organization

GETTING STARTED:
1. Double-click PhotoOrganizer.exe
2. Browser opens automatically with user-friendly interface  
3. First run downloads AI model (~150MB) - requires internet

SYSTEM REQUIREMENTS:
- Windows 10 or later
- 4GB+ RAM (8GB recommended)
- 2GB free disk space
- Internet connection (first run only)

HOW TO USE:
1. Click folder icons to select source photos and destination
2. Preview a sample to see how photos will be organized
3. Click "Run FULL (COPY)" to safely organize photos
4. App closes automatically when you close the browser

CUSTOM CATEGORIES:
- Advanced users can edit example-labels.json
- Restart app to use custom categories

That's it! No technical knowledge needed.
For developers: see README.md for source code details.
"""
    
    with open(release_dir / "USAGE.txt", "w", encoding="utf-8") as f:
        f.write(usage_content)
    print("[OK] Created: USAGE.txt")
    
    # Step 7: Calculate sizes
    total_size = sum(f.stat().st_size for f in release_dir.rglob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    
    print()
    print("=" * 50)
    print("RELEASE PACKAGE CREATED!")
    print("=" * 50)
    print(f"Location: {release_dir.absolute()}")
    print(f"Total size: {total_size_mb:.1f} MB")
    print()
    print("Contents:")
    for item in sorted(release_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name:<25} ({size_mb:.1f} MB)")
    print()
    print("NEXT STEPS:")
    print("1. Test both executables on a clean system")
    print("2. Zip the PhotoOrganizer-Release folder")
    print("3. Upload to GitHub releases")
    print()
    print("Distribution ready!")

if __name__ == "__main__":
    main()