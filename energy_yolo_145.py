from pathlib import Path
import csv
import time
import sys

import numpy as np
import torch
from PIL import Image

from codecarbon import EmissionsTracker

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression

ORIGINAL_DIR = ROOT / "videos_5" / "input_png"
RECONSTRUCTED_DIR = ROOT / "videos_5" / "results" / "base_1_145"
WEIGHTS = ROOT / "yolov5s.pt"

IMAGE_SIZE = 640
CONF_THRES = 0.25
IOU_NMS_THRES = 0.45
CLASSES = [0, 2]

OUTPUT_DIR = Path(__file__).resolve().parent / "energy_results" / "yolo_145"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


def get_images(folder):
    images = sorted(folder.glob("*.png"))
    if len(images) != 145:
        raise ValueError(
            f"{folder} : 145 images attendues, {len(images)} trouvees."
        )
    return images


def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image_rgb = np.asarray(image)
    letterboxed, _, _ = letterbox(
        image_rgb, new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False
    )
    tensor = (
        torch.from_numpy(letterboxed)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .unsqueeze(0) / 255.0
    )
    return tensor


def measured_inference(model, tensor):
    with torch.inference_mode():
        prediction = model(tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

        detections = non_max_suppression(
            prediction,
            CONF_THRES,
            IOU_NMS_THRES,
            classes=CLASSES,
        )[0]

    return detections


def measure_folder(model, folder, mode):
    images = get_images(folder)

    print(f"\nPreparation : {mode}")
    tensors = [prepare_image(path) for path in images]

    # Echauffement hors mesure
    print("Echauffement du modele...")
    for tensor in tensors[:3]:
        measured_inference(model, tensor)

    tracker = EmissionsTracker(
        project_name=f"YOLO_{mode}",
        output_dir=str(OUTPUT_DIR),
        output_file=f"{mode}.csv",
        measure_power_secs=1,
        log_level="error",
    )

    print(f"Debut de la mesure : {mode}")
    tracker.start()
    start = time.perf_counter()

    total_detections = 0
    for index, tensor in enumerate(tensors, start=1):
        detections = measured_inference(model, tensor)
        total_detections += len(detections)

        if index % 25 == 0:
            print(f"{mode} : {index}/145 images")

    elapsed = time.perf_counter() - start
    emissions = tracker.stop()
    data = tracker.final_emissions_data

    energy_wh = data.energy_consumed * 1000

    result = {
        "mode": mode,
        "images": len(images),
        "detections": total_detections,
        "duration_s": elapsed,
        "energy_Wh": energy_wh,
        "energy_mWh_per_image": energy_wh * 1000 / len(images),
        "emissions_kg_CO2eq": emissions,
    }

    print(f"\nResultat {mode}")
    for key, value in result.items():
        print(f"{key} : {value}")

    return result


def main():
    for path in (WEIGHTS, ORIGINAL_DIR, RECONSTRUCTED_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    print("Chargement de YOLOv5s...")
    model = DetectMultiBackend(
        weights=str(WEIGHTS),
        device=DEVICE,
        dnn=False,
        fp16=False,
    )
    model.eval()

    original = measure_folder(model, ORIGINAL_DIR, "original")
    reconstructed = measure_folder(
        model, RECONSTRUCTED_DIR, "reconstructed"
    )

    csv_path = OUTPUT_DIR / "energy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(original.keys()))
        writer.writeheader()
        writer.writerows([original, reconstructed])

    print("\nMesures terminees.")
    print("Resultats :", OUTPUT_DIR)
    print("Resume :", csv_path)


if __name__ == "__main__":
    main()
