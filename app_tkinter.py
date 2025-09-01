#!/usr/bin/env python3
"""
Tkinter UI for Photo Organizer
A native desktop interface for AI-powered photo organization.
"""

import os
import sys
import json
import subprocess
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

# Add parent directory to path for imports when running from source
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

try:
    from PIL import Image, ImageTk
    import torch
    import open_clip
    from default_labels import DEFAULT_LABELS
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


class PhotoOrganizerApp:
    """Main application class for Photo Organizer with Tkinter UI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Organizer - AI-Powered Photo Sorting")
        self.root.geometry("1200x800")
        
        # Configure app icon if available
        try:
            if sys.platform.startswith('win'):
                self.root.iconbitmap(default='icon.ico')
        except:
            pass
        
        # Instance variables
        self.settings = self.load_settings()
        self.preview_images = []
        self.process_thread = None
        self.process_queue = queue.Queue()
        self.is_processing = False
        
        # Setup UI
        self.setup_styles()
        self.create_menu()
        self.create_main_layout()
        self.create_status_bar()
        
        # Load saved settings into UI
        self.load_settings_to_ui()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start queue processor for subprocess output
        self.process_queue_messages()
    
    def setup_styles(self):
        """Configure ttk styles for better appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Section.TLabelframe', borderwidth=2, relief='groove')
        style.configure('Section.TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')
    
    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Settings...", command=self.load_settings_file)
        file_menu.add_command(label="Save Settings...", command=self.save_settings_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Undo Last Sort", command=self.undo_last_sort)
        tools_menu.add_command(label="Edit Labels", command=self.edit_labels)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_guide)
    
    def create_main_layout(self):
        """Create the main application layout."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Settings
        self.create_settings_panel(main_frame)
        
        # Right panel - Preview and Output
        self.create_preview_panel(main_frame)
    
    def create_settings_panel(self, parent):
        """Create the settings panel on the left side."""
        settings_frame = ttk.Frame(parent, padding="5")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Title
        title_label = ttk.Label(settings_frame, text="Settings", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        current_row = 1
        
        # Folders Section
        folders_frame = ttk.LabelFrame(settings_frame, text="Folders", padding="10", style='Section.TLabelframe')
        folders_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Source folder
        ttk.Label(folders_frame, text="Source:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.source_var = tk.StringVar(value=self.settings.get('source', ''))
        self.source_entry = ttk.Entry(folders_frame, textvariable=self.source_var, width=30)
        self.source_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(folders_frame, text="Browse...", command=self.browse_source).grid(row=0, column=2, padx=5)
        
        # Destination folder
        ttk.Label(folders_frame, text="Destination:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dest_var = tk.StringVar(value=self.settings.get('destination', ''))
        self.dest_entry = ttk.Entry(folders_frame, textvariable=self.dest_var, width=30)
        self.dest_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(folders_frame, text="Browse...", command=self.browse_destination).grid(row=1, column=2, padx=5)
        
        folders_frame.columnconfigure(1, weight=1)
        
        # Classification Section
        class_frame = ttk.LabelFrame(settings_frame, text="Classification", padding="10", style='Section.TLabelframe')
        class_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Labels configuration
        self.use_custom_labels = tk.BooleanVar(value=self.settings.get('use_custom_labels', False))
        ttk.Checkbutton(class_frame, text="Use custom labels file", 
                       variable=self.use_custom_labels,
                       command=self.toggle_custom_labels).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(class_frame, text="Labels file:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.labels_var = tk.StringVar(value=self.settings.get('labels_file', 'labels.json'))
        self.labels_entry = ttk.Entry(class_frame, textvariable=self.labels_var, width=25, state='disabled')
        self.labels_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Decision strategy
        ttk.Label(class_frame, text="Decision:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.decision_var = tk.StringVar(value=self.settings.get('decision', 'margin'))
        decision_combo = ttk.Combobox(class_frame, textvariable=self.decision_var, 
                                     values=['margin', 'ratio', 'threshold', 'always-top1'],
                                     state='readonly', width=23)
        decision_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Aggregation
        ttk.Label(class_frame, text="Aggregation:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.agg_var = tk.StringVar(value=self.settings.get('aggregation', 'max'))
        agg_combo = ttk.Combobox(class_frame, textvariable=self.agg_var,
                                values=['max', 'mean', 'sum'],
                                state='readonly', width=23)
        agg_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        
        class_frame.columnconfigure(1, weight=1)
        
        # Thresholds Section
        thresh_frame = ttk.LabelFrame(settings_frame, text="Thresholds", padding="10", style='Section.TLabelframe')
        thresh_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Margin slider
        ttk.Label(thresh_frame, text="Margin:").grid(row=0, column=0, sticky=tk.W)
        self.margin_var = tk.DoubleVar(value=self.settings.get('margin', 0.008))
        self.margin_label = ttk.Label(thresh_frame, text=f"{self.margin_var.get():.3f}")
        self.margin_label.grid(row=0, column=2, padx=5)
        margin_scale = ttk.Scale(thresh_frame, from_=0.0, to=0.03, 
                                variable=self.margin_var, orient=tk.HORIZONTAL,
                                command=lambda v: self.margin_label.config(text=f"{float(v):.3f}"))
        margin_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Ratio slider
        ttk.Label(thresh_frame, text="Ratio:").grid(row=1, column=0, sticky=tk.W)
        self.ratio_var = tk.DoubleVar(value=self.settings.get('ratio', 1.06))
        self.ratio_label = ttk.Label(thresh_frame, text=f"{self.ratio_var.get():.2f}")
        self.ratio_label.grid(row=1, column=2, padx=5)
        ratio_scale = ttk.Scale(thresh_frame, from_=1.0, to=1.3,
                               variable=self.ratio_var, orient=tk.HORIZONTAL,
                               command=lambda v: self.ratio_label.config(text=f"{float(v):.2f}"))
        ratio_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Threshold slider
        ttk.Label(thresh_frame, text="Threshold:").grid(row=2, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=self.settings.get('threshold', 0.10))
        self.threshold_label = ttk.Label(thresh_frame, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.grid(row=2, column=2, padx=5)
        threshold_scale = ttk.Scale(thresh_frame, from_=0.05, to=0.4,
                                  variable=self.threshold_var, orient=tk.HORIZONTAL,
                                  command=lambda v: self.threshold_label.config(text=f"{float(v):.2f}"))
        threshold_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        thresh_frame.columnconfigure(1, weight=1)
        
        # Options Section
        options_frame = ttk.LabelFrame(settings_frame, text="Options", padding="10", style='Section.TLabelframe')
        options_frame.grid(row=current_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Checkboxes
        self.copy_mode = tk.BooleanVar(value=self.settings.get('copy_mode', False))
        ttk.Checkbutton(options_frame, text="Copy files (don't move)", 
                       variable=self.copy_mode).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.dedupe = tk.BooleanVar(value=self.settings.get('dedupe', False))
        ttk.Checkbutton(options_frame, text="Detect duplicates", 
                       variable=self.dedupe).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.date_folders = tk.BooleanVar(value=self.settings.get('date_folders', False))
        ttk.Checkbutton(options_frame, text="Organize by date (YYYY/MM)", 
                       variable=self.date_folders).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.rich_prompts = tk.BooleanVar(value=self.settings.get('rich_prompts', True))
        ttk.Checkbutton(options_frame, text="Use rich prompts", 
                       variable=self.rich_prompts).grid(row=3, column=0, sticky=tk.W, pady=2)
        
        self.ignore_weights = tk.BooleanVar(value=self.settings.get('ignore_weights', True))
        ttk.Checkbutton(options_frame, text="Ignore label weights", 
                       variable=self.ignore_weights).grid(row=4, column=0, sticky=tk.W, pady=2)
        
        # Top-K selection
        ttk.Label(options_frame, text="Top categories:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.topk_var = tk.IntVar(value=self.settings.get('topk', 1))
        topk_spin = ttk.Spinbox(options_frame, from_=1, to=3, textvariable=self.topk_var, width=10)
        topk_spin.grid(row=5, column=1, sticky=tk.W, pady=2)
        
        # Action Buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=current_row, column=0, columnspan=2, pady=10)
        
        self.preview_btn = ttk.Button(button_frame, text="🔍 Preview (Dry Run)", 
                                     command=self.preview_sort)
        self.preview_btn.grid(row=0, column=0, padx=5)
        
        self.run_btn = ttk.Button(button_frame, text="▶ Run Full Sort", 
                                 command=self.run_sort)
        self.run_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop", 
                                  command=self.stop_sort, state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=5)
    
    def create_preview_panel(self, parent):
        """Create the preview and output panel on the right side."""
        preview_frame = ttk.Frame(parent, padding="5")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(preview_frame, text="Preview & Progress", style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(preview_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Preview tab
        self.preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_tab, text="Image Preview")
        
        # Create scrollable canvas for preview images
        self.create_preview_canvas()
        
        # Output tab
        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.output_tab, text="Output Log")
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(self.output_tab, wrap=tk.WORD, width=80, height=30)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress bar
        self.progress_frame = ttk.Frame(preview_frame)
        self.progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.pack(side=tk.LEFT, padx=5)
    
    def create_preview_canvas(self):
        """Create scrollable canvas for image previews."""
        # Create canvas and scrollbar
        canvas_frame = ttk.Frame(self.preview_tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        
        self.preview_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create frame inside canvas for content
        self.preview_content = ttk.Frame(self.preview_canvas)
        self.preview_canvas.create_window((0, 0), window=self.preview_content, anchor=tk.NW)
        
        # Bind resize event
        self.preview_content.bind('<Configure>', lambda e: self.preview_canvas.configure(
            scrollregion=self.preview_canvas.bbox('all')))
    
    def create_status_bar(self):
        """Create status bar at bottom of window."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def toggle_custom_labels(self):
        """Toggle custom labels file input."""
        if self.use_custom_labels.get():
            self.labels_entry.config(state='normal')
        else:
            self.labels_entry.config(state='disabled')
    
    def browse_source(self):
        """Open dialog to select source folder."""
        folder = filedialog.askdirectory(title="Select Source Folder", 
                                        initialdir=self.source_var.get() or os.getcwd())
        if folder:
            self.source_var.set(folder)
            self.save_settings()
    
    def browse_destination(self):
        """Open dialog to select destination folder."""
        folder = filedialog.askdirectory(title="Select Destination Folder",
                                        initialdir=self.dest_var.get() or os.getcwd())
        if folder:
            self.dest_var.set(folder)
            self.save_settings()
    
    def load_settings(self):
        """Load settings from JSON file."""
        settings_file = Path.home() / '.photo_organizer' / 'settings.json'
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self.get_default_settings()
    
    def get_default_settings(self):
        """Get default settings."""
        return {
            'source': str(Path.cwd()),
            'destination': str(Path.cwd() / 'sorted'),
            'use_custom_labels': False,
            'labels_file': 'labels.json',
            'decision': 'margin',
            'aggregation': 'max',
            'margin': 0.008,
            'ratio': 1.06,
            'threshold': 0.10,
            'copy_mode': False,
            'dedupe': False,
            'date_folders': False,
            'rich_prompts': True,
            'ignore_weights': True,
            'topk': 1
        }
    
    def save_settings(self):
        """Save current settings to JSON file."""
        settings = {
            'source': self.source_var.get(),
            'destination': self.dest_var.get(),
            'use_custom_labels': self.use_custom_labels.get(),
            'labels_file': self.labels_var.get(),
            'decision': self.decision_var.get(),
            'aggregation': self.agg_var.get(),
            'margin': self.margin_var.get(),
            'ratio': self.ratio_var.get(),
            'threshold': self.threshold_var.get(),
            'copy_mode': self.copy_mode.get(),
            'dedupe': self.dedupe.get(),
            'date_folders': self.date_folders.get(),
            'rich_prompts': self.rich_prompts.get(),
            'ignore_weights': self.ignore_weights.get(),
            'topk': self.topk_var.get()
        }
        
        settings_dir = Path.home() / '.photo_organizer'
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / 'settings.json'
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            self.settings = settings
        except Exception as e:
            self.log_message(f"Error saving settings: {e}", 'error')
    
    def load_settings_to_ui(self):
        """Load saved settings into UI controls."""
        # Settings are already loaded via StringVar/BooleanVar/etc initialization
        self.toggle_custom_labels()  # Ensure labels entry state is correct
    
    def load_settings_file(self):
        """Load settings from a selected file."""
        file_path = filedialog.askopenfilename(
            title="Load Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                self.apply_settings(settings)
                self.log_message("Settings loaded successfully", 'success')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {e}")
    
    def save_settings_file(self):
        """Save current settings to a selected file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.save_settings()
            try:
                # Copy from default location to selected location
                settings_file = Path.home() / '.photo_organizer' / 'settings.json'
                with open(settings_file, 'r') as src:
                    with open(file_path, 'w') as dst:
                        dst.write(src.read())
                self.log_message(f"Settings saved to {file_path}", 'success')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def apply_settings(self, settings):
        """Apply loaded settings to UI."""
        self.source_var.set(settings.get('source', ''))
        self.dest_var.set(settings.get('destination', ''))
        self.use_custom_labels.set(settings.get('use_custom_labels', False))
        self.labels_var.set(settings.get('labels_file', 'labels.json'))
        self.decision_var.set(settings.get('decision', 'margin'))
        self.agg_var.set(settings.get('aggregation', 'max'))
        self.margin_var.set(settings.get('margin', 0.008))
        self.ratio_var.set(settings.get('ratio', 1.06))
        self.threshold_var.set(settings.get('threshold', 0.10))
        self.copy_mode.set(settings.get('copy_mode', False))
        self.dedupe.set(settings.get('dedupe', False))
        self.date_folders.set(settings.get('date_folders', False))
        self.rich_prompts.set(settings.get('rich_prompts', True))
        self.ignore_weights.set(settings.get('ignore_weights', True))
        self.topk_var.set(settings.get('topk', 1))
        
        # Update UI state
        self.toggle_custom_labels()
        self.settings = settings
    
    def build_command(self, dry_run=False):
        """Build command line arguments for photo_sorter.py."""
        cmd = [sys.executable, "photo_sorter.py",
               "--src", self.source_var.get(),
               "--dst", self.dest_var.get(),
               "--agg", self.agg_var.get(),
               "--decision", self.decision_var.get(),
               "--margin", str(self.margin_var.get()),
               "--ratio", str(self.ratio_var.get()),
               "--threshold", str(self.threshold_var.get()),
               "--topk", str(self.topk_var.get())]
        
        if self.use_custom_labels.get() and self.labels_var.get():
            cmd.extend(["--labels", self.labels_var.get()])
        
        if self.copy_mode.get():
            cmd.append("--copy")
        if self.dedupe.get():
            cmd.append("--dedupe")
        if self.date_folders.get():
            cmd.append("--date-folders")
        if self.rich_prompts.get():
            cmd.append("--rich-prompts")
        if self.ignore_weights.get():
            cmd.append("--ignore-weights")
        
        if dry_run:
            cmd.append("--dry-run")
            cmd.append("--debug-scores")
        
        return cmd
    
    def preview_sort(self):
        """Run preview/dry-run to show what would happen."""
        if not self.validate_inputs():
            return
        
        self.clear_preview()
        self.log_message("Running preview...", 'info')
        # Don't switch tabs immediately - let the preview images appear
        
        # Build and run command
        cmd = self.build_command(dry_run=True)
        self.run_subprocess(cmd, is_preview=True)
    
    def run_sort(self):
        """Run the actual photo sorting."""
        if not self.validate_inputs():
            return
        
        # Confirm with user
        mode = "COPY" if self.copy_mode.get() else "MOVE"
        result = messagebox.askyesno(
            "Confirm Sort",
            f"This will {mode} photos from:\n{self.source_var.get()}\n\nTo:\n{self.dest_var.get()}\n\nContinue?"
        )
        
        if not result:
            return
        
        self.clear_preview()
        self.log_message(f"Starting photo sort ({mode} mode)...", 'info')
        self.notebook.select(self.output_tab)
        
        # Build and run command
        cmd = self.build_command(dry_run=False)
        self.run_subprocess(cmd, is_preview=False)
    
    def run_subprocess(self, cmd, is_preview=False):
        """Run photo_sorter.py as subprocess with output capture."""
        if self.is_processing:
            messagebox.showwarning("Processing", "Already processing. Please wait or stop current task.")
            return
        
        self.is_processing = True
        self.preview_btn.config(state='disabled')
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_var.set(0)
        self.progress_label.config(text="Processing...")
        
        def run_process():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in self.process.stdout:
                    self.process_queue.put(('output', line.strip()))
                
                self.process.wait()
                
                if self.process.returncode == 0:
                    self.process_queue.put(('complete', 'success'))
                else:
                    self.process_queue.put(('complete', f'error:{self.process.returncode}'))
                    
            except Exception as e:
                self.process_queue.put(('error', str(e)))
            finally:
                self.process = None
        
        self.process_thread = threading.Thread(target=run_process, daemon=True)
        self.process_thread.start()
    
    def stop_sort(self):
        """Stop the current sorting process."""
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
            self.log_message("Stopping process...", 'warning')
    
    def process_queue_messages(self):
        """Process messages from subprocess queue."""
        try:
            while True:
                msg_type, msg_data = self.process_queue.get_nowait()
                
                if msg_type == 'output':
                    self.log_message(msg_data)
                    # Parse progress from output
                    if 'Processing:' in msg_data and '/' in msg_data:
                        try:
                            parts = msg_data.split('Processing:')[1].strip().split('/')
                            current = int(parts[0])
                            total = int(parts[1].split()[0])
                            progress = int((current / total) * 100)
                            self.progress_var.set(progress)
                            self.progress_label.config(text=f"{current}/{total}")
                        except:
                            pass
                    
                    # Parse preview images from dry-run output
                    if '[MOVE]' in msg_data or '[COPY]' in msg_data or '[DUP]' in msg_data:
                        try:
                            # Extract source and destination paths
                            # Format: [ACTION] source -> destination
                            parts = msg_data.split(' -> ')
                            if len(parts) == 2:
                                source_part = parts[0].split('] ', 1)[1] if '] ' in parts[0] else parts[0]
                                dest_part = parts[1].strip()
                                self.add_preview_image(source_part, dest_part, msg_data)
                        except:
                            pass
                            
                elif msg_type == 'complete':
                    self.is_processing = False
                    self.preview_btn.config(state='normal')
                    self.run_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    
                    if msg_data == 'success':
                        self.progress_var.set(100)
                        self.progress_label.config(text="Complete!")
                        self.log_message("Processing completed successfully!", 'success')
                        messagebox.showinfo("Complete", "Photo sorting completed successfully!")
                    else:
                        self.progress_label.config(text="Failed")
                        self.log_message(f"Processing failed: {msg_data}", 'error')
                        
                elif msg_type == 'error':
                    self.log_message(f"Error: {msg_data}", 'error')
                    
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.process_queue_messages)
    
    def validate_inputs(self):
        """Validate user inputs before processing."""
        if not self.source_var.get():
            messagebox.showerror("Error", "Please select a source folder")
            return False
            
        if not self.dest_var.get():
            messagebox.showerror("Error", "Please select a destination folder")
            return False
            
        src_path = Path(self.source_var.get())
        if not src_path.exists():
            messagebox.showerror("Error", f"Source folder does not exist:\n{src_path}")
            return False
            
        if self.use_custom_labels.get():
            labels_path = Path(self.labels_var.get())
            if not labels_path.exists():
                messagebox.showerror("Error", f"Labels file not found:\n{labels_path}")
                return False
                
        return True
    
    def clear_preview(self):
        """Clear the preview panel."""
        for widget in self.preview_content.winfo_children():
            widget.destroy()
        self.preview_images.clear()
    
    def add_preview_image(self, source_path, dest_path, action_text):
        """Add an image to the preview panel."""
        try:
            # Switch to preview tab if not already there
            if self.notebook.index("current") != 0:
                self.notebook.select(self.preview_tab)
            
            # Create a frame for this image preview
            preview_frame = ttk.Frame(self.preview_content, relief=tk.RIDGE, borderwidth=1)
            preview_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Try to load and display thumbnail
            img_path = Path(source_path.strip())
            if img_path.exists() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                try:
                    # Load image and create thumbnail
                    img = Image.open(img_path)
                    img.thumbnail((100, 100), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Keep a reference to prevent garbage collection
                    self.preview_images.append(photo)
                    
                    # Display image
                    img_label = ttk.Label(preview_frame, image=photo)
                    img_label.pack(side=tk.LEFT, padx=5, pady=5)
                except Exception as e:
                    # If image can't be loaded, show placeholder
                    placeholder = ttk.Label(preview_frame, text="📷", font=('Segoe UI', 24))
                    placeholder.pack(side=tk.LEFT, padx=5, pady=5)
            else:
                # Show icon for non-image files
                icon_label = ttk.Label(preview_frame, text="📄", font=('Segoe UI', 24))
                icon_label.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Add text info
            info_frame = ttk.Frame(preview_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Extract action type from the message
            action = "MOVE"
            if "[COPY]" in action_text:
                action = "COPY"
            elif "[DUP]" in action_text:
                action = "DUPLICATE"
            
            # Extract label from destination path
            label = ""
            if action == "DUPLICATE":
                label = "duplicates"
            else:
                # Parse the destination path to find the label
                # Structure is: destination_folder/LABEL/[YYYY/MM/]filename
                dest_path_obj = Path(dest_path)
                dest_base = Path(self.dest_var.get()).name  # Get just the folder name, not full path
                
                # Find the label by looking at path parts after the destination folder
                parts = dest_path_obj.parts
                for i, part in enumerate(parts):
                    if part == dest_base and i + 1 < len(parts):
                        # The next part after destination folder is the label
                        label = parts[i + 1]
                        break
                
                # If we couldn't find it by folder name matching, try relative path approach
                if not label:
                    try:
                        # Get relative path from destination to file
                        rel_path = dest_path_obj.relative_to(self.dest_var.get())
                        if rel_path.parts:
                            label = rel_path.parts[0]  # First folder is the label
                    except:
                        pass
            
            # Show file info
            ttk.Label(info_frame, text=f"Action: {action}", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
            if label:
                ttk.Label(info_frame, text=f"Label: {label}", font=('Segoe UI', 9, 'italic'), foreground='blue').pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"From: {Path(source_path).name}", font=('Segoe UI', 9)).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"To: {dest_path}", font=('Segoe UI', 8), foreground='gray').pack(anchor=tk.W)
            
            # Update canvas scroll region
            self.preview_content.update_idletasks()
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox('all'))
            
        except Exception as e:
            # Silently fail if preview can't be added
            pass
    
    def log_message(self, message, level='info'):
        """Add message to output log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        
        self.output_text.insert(tk.END, formatted_msg)
        self.output_text.see(tk.END)
        
        # Update status bar for important messages
        if level in ['error', 'warning', 'success']:
            self.status_bar.config(text=message)
    
    def undo_last_sort(self):
        """Open undo dialog (placeholder for undo functionality)."""
        messagebox.showinfo("Undo", "Undo functionality requires an undo log from a previous sort.\nRun 'python undo.py' from command line if available.")
    
    def edit_labels(self):
        """Open labels editor dialog."""
        editor = LabelsEditor(self.root, self.labels_var.get())
        self.root.wait_window(editor.dialog)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Photo Organizer
AI-Powered Photo Sorting Tool

Version: 1.0 (Tkinter Edition)

Uses OpenCLIP for zero-shot image classification
to automatically organize your photo collection.

© 2024 - Weekend MVP Project"""
        messagebox.showinfo("About Photo Organizer", about_text)
    
    def show_guide(self):
        """Show user guide."""
        guide_text = """Quick Start Guide:

1. Select Source Folder - Where your unsorted photos are
2. Select Destination Folder - Where sorted photos will go
3. Choose Classification Settings:
   - Decision: How to determine photo categories
   - Thresholds: Fine-tune classification sensitivity
   
4. Preview (Dry Run) - See what would happen without moving files
5. Run Full Sort - Actually organize your photos

Tips:
- Use Copy mode for safety on first run
- Enable duplicate detection to find similar photos
- Date folders organize by EXIF date (YYYY/MM)
- Check output log for detailed progress"""
        
        messagebox.showinfo("User Guide", guide_text)
    
    def on_closing(self):
        """Handle window close event."""
        if self.is_processing:
            result = messagebox.askyesno("Confirm Exit", 
                                        "Processing is in progress. Stop and exit?")
            if result:
                self.stop_sort()
                self.root.after(500, self.root.destroy)
        else:
            self.save_settings()
            self.root.destroy()


class LabelsEditor:
    """Simple labels file editor dialog."""
    
    def __init__(self, parent, labels_file):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Edit Labels")
        self.dialog.geometry("600x400")
        self.labels_file = labels_file
        
        # Text editor
        self.text = scrolledtext.ScrolledText(self.dialog, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Load", command=self.load_labels).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_labels).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Load current labels
        self.load_labels()
    
    def load_labels(self):
        """Load labels file into editor."""
        try:
            labels_path = Path(self.labels_file)
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    content = f.read()
            else:
                # Load default labels as template
                content = json.dumps(DEFAULT_LABELS, indent=2)
            
            self.text.delete(1.0, tk.END)
            self.text.insert(1.0, content)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load labels: {e}")
    
    def save_labels(self):
        """Save edited labels to file."""
        try:
            content = self.text.get(1.0, tk.END).strip()
            # Validate JSON
            json.loads(content)
            
            with open(self.labels_file, 'w') as f:
                f.write(content)
            
            messagebox.showinfo("Success", "Labels saved successfully!")
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON", f"JSON syntax error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {e}")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = PhotoOrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()