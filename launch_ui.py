#!/usr/bin/env python3
"""
Simple launcher for Photo Organizer UI.
This script provides a clean way to launch the Streamlit interface.
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    """Launch the Streamlit UI and open browser."""
    print("🚀 Starting Photo Organizer...")
    print("📷 AI-powered photo organization tool")
    print()
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "app_streamlit.py"
    
    if not app_file.exists():
        print("❌ Error: app_streamlit.py not found!")
        print(f"Expected location: {app_file}")
        input("Press Enter to exit...")
        return
    
    try:
        print("🔧 Launching web interface...")
        print("💡 Tip: The app will auto-close when you close your browser")
        print()
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), 
               "--browser.gatherUsageStats", "false"]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using Photo Organizer.")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Try running directly: streamlit run app_streamlit.py")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()