@echo off
echo Photo Organizer - AI-powered photo organization
echo.
echo Choose how to run:
echo 1) Web UI (recommended for beginners)
echo 2) Command Line (for advanced users)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting web interface...
    python launch_ui.py
) else if "%choice%"=="2" (
    echo.
    echo Command line mode. For help: python photo_sorter.py --help
    echo Example: python photo_sorter.py --src "C:\Photos\Unsorted" --dst "C:\Photos\Sorted" --dry-run
    echo.
    cmd /k
) else (
    echo Invalid choice. Please run again and choose 1 or 2.
    pause
)