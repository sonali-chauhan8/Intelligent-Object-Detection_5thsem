from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "coco" / "subset500"

model = YOLO(str(ROOT / "yolo11n.pt"))

lines = [
    f"path: {DATASET.as_posix()}",
    "train: images/val2017",
    "val: images/val2017",
    "names:"
]

for class_id, name in model.names.items():
    lines.append(f"  {class_id}: '{name}'")

yaml_path = DATASET / "coco_subset.yaml"
yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("YAML regenerated successfully!")
print(yaml_path.read_text(encoding="utf-8")[:300])
