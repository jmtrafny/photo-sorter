import os
import shutil
import argparse

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

def flatten_images(src, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)

    for root, _, files in os.walk(src):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest, file)

                # Avoid overwriting if same filename exists
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(dest, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten image hierarchy into one folder")
    parser.add_argument("--src", required=True, help="Source folder")
    parser.add_argument("--dst", required=True, help="Destination folder")

    args = parser.parse_args()
    flatten_images(args.src, args.dst)
