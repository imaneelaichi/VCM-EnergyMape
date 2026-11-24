"""
Comparaison de la dÃ©tection YOLOv5 sur les images originales et reconstruites
Projet VCM - Base Layer

Ã€ placer dans la racine du dÃ©pÃ´t svc_code, au mÃªme niveau que :
  - yolov5s.pt
  - models/
  - utils/
  - videos_5/

ExÃ©cution (environnement svc_env activÃ©) :
  python compare_detection.py

Les chemins et paramÃ¨tres peuvent Ãªtre modifiÃ©s dans la section CONFIGURATION.
"""

from pathlib import Path
import csv
import time
import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.metrics import ap_per_class, box_iou


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

ORIGINAL_DIR = ROOT / "videos_5" / "input_png"
RECONSTRUCTED_DIR = ROOT / "videos_5" / "results" / "base_1_145"
LABELS_DIR = ROOT / "videos_5" / "labels"

WEIGHTS = ROOT / "yolov5s.pt"
DATA_YAML = ROOT / "data" / "hevc_class_b_coded.yaml"

OUTPUT_DIR = ROOT / "videos_5" / "results" / "detection_comparison_145"
ORIGINAL_OUTPUT_DIR = OUTPUT_DIR / "original"
RECONSTRUCTED_OUTPUT_DIR = OUTPUT_DIR / "reconstructed"

IMAGE_SIZE = 640
CONF_THRES = 0.25
IOU_NMS_THRES = 0.45
IOU_METRIC_THRESHOLDS = np.linspace(0.50, 0.95, 10)

# Le jeu de donnÃ©es contient uniquement les classes COCO :
# 0 = person, 2 = car.
CLASSES_TO_EVALUATE = {0: "person", 2: "car"}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# OUTILS
# ============================================================

def frame_number(path: Path):
    """Extrait le numÃ©ro d'une image : frame_001.png -> 1."""
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def find_images(folder: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions],
        key=lambda p: frame_number(p) if frame_number(p) is not None else 10**9
    )


def load_yolo_labels(label_path: Path, width: int, height: int):
    """
    Lit les annotations YOLO normalisÃ©es :
    class x_center y_center width height
    et les convertit en xyxy en pixels de l'image originale.
    """
    targets = []
    if not label_path.exists():
        raise FileNotFoundError(f"Annotation introuvable : {label_path}")

    with label_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"Attention : ligne ignorÃ©e dans {label_path.name}, ligne {line_number}")
                continue

            cls, xc, yc, bw, bh = map(float, parts)
            cls = int(cls)

            if cls not in CLASSES_TO_EVALUATE:
                continue

            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height

            targets.append([cls, x1, y1, x2, y2])

    if not targets:
        return torch.zeros((0, 5), dtype=torch.float32)

    return torch.tensor(targets, dtype=torch.float32)


def scale_boxes_from_letterbox(boxes, ratio, pad, original_width, original_height):
    """RamÃ¨ne les boÃ®tes de l'image letterbox vers l'image originale."""
    boxes = boxes.clone()
    gain = ratio[0]
    pad_x, pad_y = pad

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain

    boxes[:, [0, 2]].clamp_(0, original_width)
    boxes[:, [1, 3]].clamp_(0, original_height)
    return boxes


def match_predictions(detections, targets):
    """
    Produit une matrice TP pour les seuils IoU 0.50 Ã  0.95.
    detections : Nx6 [x1,y1,x2,y2,confidence,class]
    targets    : Mx5 [class,x1,y1,x2,y2]
    """
    correct = np.zeros((len(detections), len(IOU_METRIC_THRESHOLDS)), dtype=bool)

    if len(detections) == 0 or len(targets) == 0:
        return correct

    target_classes = targets[:, 0]
    target_boxes = targets[:, 1:5]
    detection_classes = detections[:, 5]

    ious = box_iou(target_boxes, detections[:, :4])
    class_matches = target_classes[:, None] == detection_classes[None, :]

    for threshold_index, threshold in enumerate(IOU_METRIC_THRESHOLDS):
        candidates = torch.where((ious >= threshold) & class_matches)

        if candidates[0].numel() == 0:
            continue

        matches = torch.cat(
            (
                torch.stack(candidates, 1).float(),
                ious[candidates[0], candidates[1]][:, None],
            ),
            1,
        ).cpu().numpy()

        # PrioritÃ© aux correspondances ayant le meilleur IoU.
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        correct[matches[:, 1].astype(int), threshold_index] = True

    return correct


def draw_detections(image_rgb, detections, output_path: Path):
    """Enregistre une copie de l'image avec les boÃ®tes prÃ©dites."""
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        class_name = CLASSES_TO_EVALUATE.get(int(cls), str(int(cls)))
        label = f"{class_name} {conf:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0, y1 - 18)), label, fill="red", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


# ============================================================
# Ã‰VALUATION D'UN DOSSIER
# ============================================================

def evaluate_folder(model, image_dir: Path, annotated_dir: Path, mode_name: str):
    images = find_images(image_dir)
    if not images:
        raise FileNotFoundError(f"Aucune image trouvÃ©e dans : {image_dir}")

    stats = []
    target_class_list = []
    detection_rows = []
    inference_times = []
    total_targets = 0
    total_detections = 0

    print(f"\n===== Ã‰valuation : {mode_name} =====")
    print(f"Dossier : {image_dir}")
    print(f"Nombre d'images : {len(images)}")

    for index, image_path in enumerate(images, start=1):
        number = frame_number(image_path)
        if number is None:
            print(f"Image ignorÃ©e (numÃ©ro introuvable) : {image_path.name}")
            continue

        # Les images s'appellent frame_001.png ; les labels frame_0001.txt.
        label_path = LABELS_DIR / f"frame_{number:04d}.txt"

        image = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image)
        original_height, original_width = image_rgb.shape[:2]

        # PrÃ©traitement YOLOv5 : redimensionnement + padding.
        letterboxed, ratio, pad = letterbox(
            image_rgb, new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False
        )
        input_tensor = (
            torch.from_numpy(letterboxed)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .unsqueeze(0)
            .to(DEVICE) / 255.0
        )

        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(input_tensor)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            detections = non_max_suppression(
                prediction,
                CONF_THRES,
                IOU_NMS_THRES,
                classes=list(CLASSES_TO_EVALUATE.keys()),
            )[0]
        inference_time = time.perf_counter() - start
        inference_times.append(inference_time)
        # Copier les rÃ©sultats pour pouvoir modifier les coordonnÃ©es
        detections = detections.clone()

        if len(detections):
            detections[:, :4] = scale_boxes_from_letterbox(
                detections[:, :4], ratio, pad, original_width, original_height
            )

        targets = load_yolo_labels(label_path, original_width, original_height)
        total_targets += len(targets)
        total_detections += len(detections)

        correct = match_predictions(detections.cpu(), targets)
        stats.append(
            (
                correct,
                detections[:, 4].cpu().numpy() if len(detections) else np.array([]),
                detections[:, 5].cpu().numpy() if len(detections) else np.array([]),
                targets[:, 0].cpu().numpy() if len(targets) else np.array([]),
            )
        )

        for det in detections.cpu():
            x1, y1, x2, y2, conf, cls = det.tolist()
            detection_rows.append({
                "mode": mode_name,
                "image": image_path.name,
                "class_id": int(cls),
                "class_name": CLASSES_TO_EVALUATE[int(cls)],
                "confidence": round(conf, 6),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            })

        draw_detections(image_rgb, detections.cpu(), annotated_dir / image_path.name)

        print(
            f"[{index}/{len(images)}] {image_path.name} : "
            f"{len(detections)} dÃ©tection(s), {len(targets)} objet(s) annotÃ©(s), "
            f"{inference_time:.3f} s"
        )

    if not stats:
        raise RuntimeError(f"Aucune image valide Ã©valuÃ©e pour {mode_name}.")

    # ConcatÃ©nation des rÃ©sultats de toutes les images.
    tp = np.concatenate([s[0] for s in stats], axis=0)
    conf = np.concatenate([s[1] for s in stats], axis=0)
    pred_cls = np.concatenate([s[2] for s in stats], axis=0)
    target_cls = np.concatenate([s[3] for s in stats], axis=0)

    if len(target_cls) == 0:
        raise RuntimeError("Aucune annotation rÃ©elle trouvÃ©e : vÃ©rifiez LABELS_DIR.")

    # ap_per_class retourne P, R, F1, AP, classes selon cette version YOLOv5.
    result = ap_per_class(
        tp, conf, pred_cls, target_cls,
        plot=False,
        save_dir=str(OUTPUT_DIR),
        names=CLASSES_TO_EVALUATE,
    )

    # CompatibilitÃ© avec la signature YOLOv5 utilisÃ©e dans ce dÃ©pÃ´t.
    if len(result) == 7:
        _, _, precision, recall, f1, ap, ap_class = result
    elif len(result) == 5:
        precision, recall, f1, ap, ap_class = result
    else:
        raise RuntimeError(
            f"Format de retour ap_per_class inattendu : {len(result)} valeurs."
        )

    def first_value(values, index):
        # Selon la version de YOLOv5, P/R/F1 peuvent Ãªtre des vecteurs
        # (nc,) ou des matrices (nc, seuils_de_confiance).
        row = values[index]
        return float(row[0] if np.ndim(row) > 0 else row)

    per_class = {}
    for i, class_id in enumerate(ap_class):
        class_id = int(class_id)
        if class_id not in CLASSES_TO_EVALUATE:
            continue
        per_class[CLASSES_TO_EVALUATE[class_id]] = {
            "precision": first_value(precision, i),
            "recall": first_value(recall, i),
            "mAP50": float(ap[i, 0]),
            "mAP50_95": float(ap[i].mean()),
            "f1": first_value(f1, i),
        }

    # Moyenne des classes effectivement prÃ©sentes dans les annotations.
    evaluated_classes = [
        values for class_name, values in per_class.items()
        if np.any(target_cls == next(k for k, v in CLASSES_TO_EVALUATE.items() if v == class_name))
    ]
    if not evaluated_classes:
        raise RuntimeError("Aucune classe Ã©valuÃ©e.")

    summary = {
        "mode": mode_name,
        "images": len(stats),
        "ground_truth_objects": int(total_targets),
        "predictions": int(total_detections),
        "precision": float(np.mean([x["precision"] for x in evaluated_classes])),
        "recall": float(np.mean([x["recall"] for x in evaluated_classes])),
        "mAP50": float(np.mean([x["mAP50"] for x in evaluated_classes])),
        "mAP50_95": float(np.mean([x["mAP50_95"] for x in evaluated_classes])),
        "mean_inference_time_s": float(np.mean(inference_times)),
        "total_inference_time_s": float(np.sum(inference_times)),
    }

    return summary, per_class, detection_rows


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("PÃ©riphÃ©rique :", DEVICE)
    print("Poids YOLO :", WEIGHTS)
    print("Images originales :", ORIGINAL_DIR)
    print("Images reconstruites :", RECONSTRUCTED_DIR)
    print("Annotations :", LABELS_DIR)

    for required_path in (WEIGHTS, ORIGINAL_DIR, RECONSTRUCTED_DIR, LABELS_DIR):
        if not required_path.exists():
            raise FileNotFoundError(f"Chemin introuvable : {required_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECONSTRUCTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = DetectMultiBackend(
        weights=str(WEIGHTS),
        device=DEVICE,
        dnn=False,
        data=str(DATA_YAML) if DATA_YAML.exists() else None,
        fp16=False,
    )
    model.eval()

    original_summary, original_classes, original_detections = evaluate_folder(
        model, ORIGINAL_DIR, ORIGINAL_OUTPUT_DIR, "original"
    )
    reconstructed_summary, reconstructed_classes, reconstructed_detections = evaluate_folder(
        model, RECONSTRUCTED_DIR, RECONSTRUCTED_OUTPUT_DIR, "reconstructed"
    )

    # RÃ©sumÃ© comparatif.
    summary_rows = [original_summary, reconstructed_summary]
    write_csv(
        OUTPUT_DIR / "metrics_summary.csv",
        summary_rows,
        list(original_summary.keys()),
    )

    # MÃ©triques par classe et par version d'image.
    class_rows = []
    for mode, class_metrics in (
        ("original", original_classes),
        ("reconstructed", reconstructed_classes),
    ):
        for class_name, values in class_metrics.items():
            class_rows.append({"mode": mode, "class": class_name, **values})

    write_csv(
        OUTPUT_DIR / "metrics_per_class.csv",
        class_rows,
        ["mode", "class", "precision", "recall", "mAP50", "mAP50_95", "f1"],
    )

    # Toutes les dÃ©tections produites.
    all_detections = original_detections + reconstructed_detections
    write_csv(
        OUTPUT_DIR / "detections.csv",
        all_detections,
        ["mode", "image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
    )

    # Tableau des Ã©carts (reconstruit - original).
    comparison_rows = []
    for metric in (
        "precision", "recall", "mAP50", "mAP50_95",
        "mean_inference_time_s", "total_inference_time_s",
    ):
        comparison_rows.append({
            "metric": metric,
            "original": original_summary[metric],
            "reconstructed": reconstructed_summary[metric],
            "difference_reconstructed_minus_original":
                reconstructed_summary[metric] - original_summary[metric],
        })

    write_csv(
        OUTPUT_DIR / "comparison.csv",
        comparison_rows,
        ["metric", "original", "reconstructed", "difference_reconstructed_minus_original"],
    )

    # Copie JSON lisible des rÃ©sultats.
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "original": original_summary,
                "reconstructed": reconstructed_summary,
                "per_class": {
                    "original": original_classes,
                    "reconstructed": reconstructed_classes,
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("\n========== COMPARAISON FINALE ==========")
    print(f"{'MÃ©trique':<25}{'Original':>15}{'Reconstruit':>17}{'Ã‰cart':>15}")
    for row in comparison_rows:
        print(
            f"{row['metric']:<25}"
            f"{row['original']:>15.5f}"
            f"{row['reconstructed']:>17.5f}"
            f"{row['difference_reconstructed_minus_original']:>15.5f}"
        )

    print("\nÃ‰valuation terminÃ©e.")
    print("RÃ©sultats enregistrÃ©s dans :", OUTPUT_DIR)


if __name__ == "__main__":
    main()

