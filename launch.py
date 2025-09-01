#!/usr/bin/env python3
"""
Simple launcher for Photo Organizer with Tkinter UI.
This script provides a clean way to launch the application.
"""

import sys
import os
from pathlib import Path


def main():
    """Launch the Photo Organizer Tkinter UI."""
    print("Starting Photo Organizer...")
    print("AI-powered photo organization tool")
    print("-" * 40)
    
    # Determine if we're running from a PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running from executable
        print("Running from compiled executable")
        app_path = os.path.join(sys._MEIPASS, "app_tkinter.py")
    else:
        # Running from source
        print("Running from source code")
        script_dir = Path(__file__).parent
        app_path = script_dir / "app_tkinter.py"
    
    # Check if app file exists
    if not os.path.exists(str(app_path)):
        print(f"\n❌ Error: app_tkinter.py not found!")
        print(f"Expected location: {app_path}")
        input("\nPress Enter to exit...")
        return 1
    
    try:
        # Import and run the Tkinter app directly
        import tkinter as tk
        
        # Add the app directory to Python path
        sys.path.insert(0, str(Path(app_path).parent))
        
        # Import the app module
        from app_tkinter import PhotoOrganizerApp
        
        # Create and run the application
        root = tk.Tk()
        app = PhotoOrganizerApp(root)
        root.mainloop()
        
        print("\nPhoto Organizer closed.")
        return 0
        
    except ImportError as e:
        print(f"\n❌ Error: Missing dependency - {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        input("\nPress Enter to exit...")
        return 1
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! Thanks for using Photo Organizer.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Try running directly: python app_tkinter.py")
        input("\nPress Enter to exit...")
        return 1


if __name__ == "__main__":
    sys.exit(main())