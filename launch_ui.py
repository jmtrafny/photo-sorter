#!/usr/bin/env python3
"""
Simple launcher for Photo Organizer UI.
This script provides a clean way to launch the Streamlit interface.
"""

import subprocess
import sys
import webbrowser
import time
import os
from pathlib import Path

def main():
    """Launch the Streamlit UI and open browser."""
    print("Starting Photo Organizer...")
    print("AI-powered photo organization tool")
    print()
    
    # Get the directory containing this script or bundled app
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running from PyInstaller bundle
        print("Running from executable...")
        app_file = os.path.join(sys._MEIPASS, "app_streamlit.py")
    else:
        # Running from source
        print("Running from source...")
        script_dir = Path(__file__).parent
        app_file = script_dir / "app_streamlit.py"
    
    if not os.path.exists(str(app_file)):
        print("[ERROR] app_streamlit.py not found!")
        print(f"Expected location: {app_file}")
        input("Press Enter to exit...")
        return
    
    try:
        print("Launching web interface...")
        print("Tip: The app will auto-close when you close your browser")
        print()
        
        # Check if we're in a PyInstaller bundle
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # In PyInstaller bundle - create a batch file to run streamlit from system Python
            print("Creating launcher script...")
            
            import tempfile
            import shutil
            
            # Create a temporary directory for our launcher
            temp_dir = tempfile.mkdtemp(prefix="photo_organizer_")
            
            try:
                # Copy the app file to temp directory
                temp_app_file = os.path.join(temp_dir, "app_streamlit.py")
                shutil.copy2(app_file, temp_app_file)
                
                # Copy dependencies if they exist
                deps_files = ["default_labels.py", "photo_sorter.py"]
                for dep_file in deps_files:
                    src_path = os.path.join(sys._MEIPASS, dep_file)
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, os.path.join(temp_dir, dep_file))
                
                # Create a batch file to launch streamlit
                batch_file = os.path.join(temp_dir, "launch.bat")
                batch_content = f'''@echo off
echo Launching Photo Organizer web interface...
echo.
cd /d "{temp_dir}"
python -m streamlit run app_streamlit.py --browser.gatherUsageStats false
echo.
echo Photo Organizer has stopped.
pause
'''
                
                with open(batch_file, 'w') as f:
                    f.write(batch_content)
                
                print(f"Temporary files created in: {temp_dir}")
                print("Launching external Python process...")
                print("Note: This requires Python and Streamlit to be installed on your system.")
                print()
                
                # Run the batch file
                result = subprocess.run([batch_file], shell=True)
                
            except Exception as e:
                print(f"Failed to create launcher: {e}")
                print("Please ensure Python and Streamlit are installed on your system.")
                input("Press Enter to exit...")
            finally:
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
        else:
            # Running from source - use normal subprocess
            cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), 
                   "--browser.gatherUsageStats", "false"]
            subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nGoodbye! Thanks for using Photo Organizer.")
    except Exception as e:
        print(f"[ERROR] Error launching application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Try running directly: streamlit run app_streamlit.py")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()