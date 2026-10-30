from pathlib import Path
from ultralytics import YOLO
import json

ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "datasets" / "coco" / "subset500" / "coco_subset.yaml"

print("Loading YOLO11n model...")
model = YOLO(str(ROOT / "yolo11n.pt"))

print("Starting baseline evaluation on CPU...")

metrics = model.val(
    data=str(DATASET_YAML),
    imgsz=640,
    batch=2,
    device="cpu",
    workers=0,
    plots=True,
    project=str(ROOT / "evaluation" / "runs"),
    name="yolo11n_baseline"
)

results = {
    "model": "YOLO11n",
    "dataset": "COCO validation subset (500 images)",
    "device": "CPU",
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map)
}

results["F1"] = (
    2 * results["precision"] * results["recall"] /
    (results["precision"] + results["recall"])
    if results["precision"] + results["recall"] > 0 else 0
)

output = ROOT / "evaluation" / "yolo11n_baseline.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=4), encoding="utf-8")

print("\nBASELINE RESULTS")
for key, value in results.items():
    print(f"{key}: {value}")
print("\nSaved results to:", output)
