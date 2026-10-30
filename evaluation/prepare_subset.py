import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "coco"

IMAGE_SOURCE = DATASET / "images" / "val2017"
ANNOTATION_SOURCE = DATASET / "annotations" / "instances_val2017.json"

OUTPUT = DATASET / "subset500"
IMAGE_OUTPUT = OUTPUT / "images" / "val2017"
ANNOTATION_OUTPUT = OUTPUT / "annotations"

SAMPLE_SIZE = 500
SEED = 42

print("Loading COCO annotations...")

with open(ANNOTATION_SOURCE, "r", encoding="utf-8") as file:
    coco = json.load(file)

images = coco["images"]

if len(images) < SAMPLE_SIZE:
    raise ValueError("Not enough images in the source dataset.")

random.seed(SEED)
selected_images = random.sample(images, SAMPLE_SIZE)

selected_ids = {image["id"] for image in selected_images}

selected_annotations = [
    annotation
    for annotation in coco["annotations"]
    if annotation["image_id"] in selected_ids
]

print("Copying selected images...")

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
ANNOTATION_OUTPUT.mkdir(parents=True, exist_ok=True)

for image in selected_images:
    source = IMAGE_SOURCE / image["file_name"]
    destination = IMAGE_OUTPUT / image["file_name"]

    if not source.exists():
        raise FileNotFoundError(f"Missing image: {source}")

    shutil.copy2(source, destination)

subset = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": coco["categories"]
}

with open(
    ANNOTATION_OUTPUT / "instances_val2017.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(subset, file)

print("\nSubset preparation complete!")
print("Selected images:", len(selected_images))
print("Selected annotations:", len(selected_annotations))
print("Categories:", len(subset["categories"]))
print("Output folder:", OUTPUT)
