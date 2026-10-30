from pathlib import Path
from ultralytics import YOLO
import json
import time

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "coco" / "subset500" / "coco_subset.yaml"

model = YOLO(str(ROOT / "yolo11s.pt"))

results = {}

for confidence in [0.15, 0.25, 0.40]:
    print(f"\nTesting confidence: {confidence}")

    start = time.time()

    metrics = model.val(
        data=str(DATA),
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        conf=confidence,
        plots=False,
        project=str(ROOT / "evaluation" / "runs"),
        name=f"yolo11s_conf_{confidence}"
    )

    results[str(confidence)] = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "elapsed_seconds": round(time.time() - start, 2)
    }

output = ROOT / "evaluation" / "yolo11s_confidence_comparison.json"
output.write_text(json.dumps(results, indent=4), encoding="utf-8")

print("\nCONFIDENCE COMPARISON")
print(json.dumps(results, indent=4))
print("\nSaved to:", output)
