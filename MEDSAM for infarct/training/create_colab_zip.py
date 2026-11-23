"""
create_colab_zip.py
===================
Packs APIS_Dataset, phase_2_yolo26 annotations, and MEDSAM for infarct codebase
into a single lightweight zip file for Google Colab upload.
Excludes *.pt weights and cache files to keep the zip ~180 MB.
"""

import os
import zipfile
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIP_OUT = os.path.join(PROJECT_ROOT, "apis_finetune_colab_pack.zip")

EXCLUDE_EXTS = {".pt", ".pyc"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".idea"}

INCLUDE_TARGETS = [
    "APIS_Dataset",
    os.path.join("phase_2_yolo26", "dataset", "labels"),
    os.path.join("phase_2_yolo26", "dataset", "generation_stats.json"),
    "MEDSAM for infarct"
]

print(f"Creating Colab training package: {ZIP_OUT}")
count = 0
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zipf:
    for target in INCLUDE_TARGETS:
        full_target = os.path.join(PROJECT_ROOT, target)
        if os.path.isfile(full_target):
            zipf.write(full_target, target)
            count += 1
        elif os.path.isdir(full_target):
            for root, dirs, files in os.walk(full_target):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for f in files:
                    if any(f.endswith(ext) for ext in EXCLUDE_EXTS):
                        continue
                    file_path = os.path.join(root, f)
                    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                    zipf.write(file_path, rel_path)
                    count += 1

size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
print(f"SUCCESS! Added {count} files.")
print(f"Archive Size: {size_mb:.2f} MB")
print(f"Saved at: {ZIP_OUT}")

