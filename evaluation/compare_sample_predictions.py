from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

image_dir = ROOT / "datasets/coco/subset500/images/val2017"
output_dir = ROOT / "evaluation/predictions"

image_names = [
    "000000000776.jpg",
    "000000002431.jpg",
    "000000004395.jpg",
    "000000007818.jpg",
    "000000009914.jpg",
]

models = {
    "yolo11n": YOLO(str(ROOT / "yolo11n.pt")),
    "yolo11s": YOLO(str(ROOT / "yolo11s.pt")),
}

for model_name, model in models.items():
    (output_dir / model_name).mkdir(parents=True, exist_ok=True)

for image_name in image_names:
    image_path = image_dir / image_name

    if not image_path.exists():
        print(f"Image not found: {image_name}")
        continue

    predictions = {}

    for model_name, model in models.items():
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )[0]

        annotated = result.plot()
        predictions[model_name] = annotated

        save_path = output_dir / model_name / image_name
        cv2.imwrite(str(save_path), annotated)

    # Create side-by-side comparison
    n_image = predictions["yolo11n"]
    s_image = predictions["yolo11s"]

    cv2.putText(
        n_image, "YOLO11n", (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.putText(
        s_image, "YOLO11s", (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    comparison = cv2.hconcat([n_image, s_image])

    cv2.imwrite(
        str(output_dir / f"comparison_{image_name}"),
        comparison
    )

    print(f"Compared: {image_name}")

print("\nAll comparisons completed!")
print(f"Results saved in: {output_dir}")