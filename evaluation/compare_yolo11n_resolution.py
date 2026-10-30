from pathlib import Path
from ultralytics import YOLO
import json
import time

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "coco" / "subset500" / "coco_subset.yaml"

model = YOLO(str(ROOT / "yolo11n.pt"))

results = {}

for size in [640, 960]:
    print(f"\nEvaluating YOLO11n at image size {size}...")

    start = time.time()

    metrics = model.val(
        data=str(DATA),
        imgsz=size,
        batch=2,
        device="cpu",
        workers=0,
        plots=False,
        project=str(ROOT / "evaluation" / "runs"),
        name=f"yolo11n_imgsz_{size}"
    )

    elapsed = time.time() - start

    results[str(size)] = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "elapsed_seconds": round(elapsed, 2)
    }

output = ROOT / "evaluation" / "yolo11n_resolution_comparison.json"
output.write_text(json.dumps(results, indent=4), encoding="utf-8")

print("\nRESOLUTION COMPARISON")
print(json.dumps(results, indent=4))
print("\nSaved to:", output)
