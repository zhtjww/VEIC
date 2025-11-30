import os
import shutil
from pathlib import Path

source_dirs = ["datasets/VOC2007/JPEGImages", "datasets/VOC2012/JPEGImages"]
target_dir = "datasets/VEIC_data/VOCImages"

Path(target_dir).mkdir(parents=True, exist_ok=True)

for source_dir in source_dirs:
    if os.path.exists(source_dir):
        for filename in os.listdir(source_dir):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)

print("done")