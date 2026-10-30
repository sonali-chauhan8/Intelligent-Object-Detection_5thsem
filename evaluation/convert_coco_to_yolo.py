import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "coco" / "subset500"

ANNOTATION_FILE = DATASET / "annotations" / "instances_val2017.json"
IMAGE_DIR = DATASET / "images" / "val2017"
LABEL_DIR = DATASET / "labels" / "val2017"

print("Loading annotations and model classes...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

model = YOLO(str(ROOT / "yolo11n.pt"))
model_names = model.names

category_to_class = {
    category["id"]: next(
        class_id
        for class_id, name in model_names.items()
        if name == category["name"]
    )
    for category in coco["categories"]
}

annotations_by_image = {}

for ann in coco["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

LABEL_DIR.mkdir(parents=True, exist_ok=True)

print("Converting annotations...")

for image in coco["images"]:
    image_id = image["id"]
    width = image["width"]
    height = image["height"]

    label_path = LABEL_DIR / (Path(image["file_name"]).stem + ".txt")
    lines = []

    for ann in annotations_by_image.get(image_id, []):
        x, y, box_width, box_height = ann["bbox"]

        if box_width <= 0 or box_height <= 0:
            continue

        x_center = (x + box_width / 2) / width
        y_center = (y + box_height / 2) / height
        normalized_width = box_width / width
        normalized_height = box_height / height

        class_id = category_to_class[ann["category_id"]]

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )

    label_path.write_text("\n".join(lines), encoding="utf-8")

print("\nConversion complete!")
print("Images processed:", len(coco["images"]))
print("Label files created:", len(list(LABEL_DIR.glob("*.txt"))))
print("Classes mapped:", len(category_to_class))
