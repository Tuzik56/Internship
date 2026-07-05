"""
Единый скрипт подсчёта метрик качества для всех 5 моделей.

Считает mAP50, mAP50-95, Precision, Recall, F1 по одной и той же процедуре
для всех моделей (а не встроенными средствами каждого фреймворка по
отдельности) — иначе цифры между моделями будут не сопоставимы:
    - mAP50 / mAP50-95: pycocotools.COCOeval (стандарт COCO);
    - Precision / Recall / F1: собственный подсчёт TP/FP/FN при
      фиксированных порогах (--conf-threshold, --iou-threshold),
      усреднённый по классам (macro-average).

Можно посчитать все 5 моделей одной командой или по одной за раз — результаты
в любом случае дописываются в общие metrics_summary.json / metrics_per_class.json,
не перезаписывая то, что посчитано в предыдущих запусках.

Использование:
    python -m src.evaluation.metrics                    # все 5 моделей подряд
    python -m src.evaluation.metrics --models yolo detr  # только эти две
    python -m src.evaluation.metrics --models yolo       # только одна
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.dataset.classes import CLASS_NAMES
from src.models.model_factory import get_model

ALL_MODELS = ["faster_rcnn", "ssd", "retinanet", "detr", "yolo"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--checkpoint-tag", default="best", choices=["best", "last"])
    parser.add_argument("--images-dir", default="data/raw/kitti/training/image_2")
    parser.add_argument("--val-ann", default="data/processed/annotations/val.json")
    parser.add_argument("--checkpoints-dir", default="results/checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Инференс: у каждой модели свой API, но на выходе всех — единый формат
# COCO-детекций: {"image_id", "category_id", "bbox": [x, y, w, h], "score"}.
# category_id приведён к нумерации val.json (1..6, см. src/dataset/classes.py
# и coco_export.py) — у DETR и YOLO предсказанные классы 0-индексированы,
# поэтому для них +1.
# --------------------------------------------------------------------------

@torch.no_grad()
def detect_torchvision(model_name: str, checkpoint_path: Path, images_dir: str, val_ann: str, device) -> list[dict]:
    from torch.utils.data import DataLoader

    from src.dataset.detection_dataset import CocoDetectionDataset, collate_fn

    model = get_model(model_name, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    dataset = CocoDetectionDataset(images_dir, val_ann)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    detections = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                detections.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

    return detections


@torch.no_grad()
def detect_detr(checkpoint_dir: Path, images_dir: str, val_ann: str, device) -> list[dict]:
    from transformers import DetrForObjectDetection, DetrImageProcessor

    from src.dataset.detr_dataset import DetrCocoDataset

    processor = DetrImageProcessor.from_pretrained(checkpoint_dir)
    model = DetrForObjectDetection.from_pretrained(checkpoint_dir).to(device).eval()

    dataset = DetrCocoDataset(images_dir, val_ann)

    detections = []
    for image, coco_annotation in dataset:
        image_id = coco_annotation["image_id"]

        encoding = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**encoding)

        # (height, width) исходного изображения — чтобы боксы вернулись
        # в пиксельных координатах оригинала, а не resize-версии
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]

        for (x1, y1, x2, y2), score, label in zip(
            results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist()
        ):
            detections.append({
                "image_id": image_id,
                "category_id": int(label) + 1,  # DETR: 0-индексирован
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    return detections


def detect_yolo(checkpoint_path: Path, images_dir: str, coco_gt: COCO, device) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))

    image_infos = list(coco_gt.dataset["images"])
    image_paths = [str(Path(images_dir) / info["file_name"]) for info in image_infos]

    batch_size = 16
    detections = []
    for start in range(0, len(image_paths), batch_size):
        chunk_infos = image_infos[start : start + batch_size]
        chunk_paths = image_paths[start : start + batch_size]

        # передавать сразу весь image_paths (1497 путей) в predict() нельзя:
        # список str-источников ultralytics прогоняет через LoadPilAndNumpy,
        # у которого self.bs = len(source) жёстко, независимо от аргумента
        # batch — весь список схлопывается в один forward-батч и упирается
        # в OOM ещё до первого результата. Поэтому бьём на чанки вручную.
        results = model.predict(chunk_paths, conf=0.05, device=device, verbose=False)

        for image_info, result in zip(chunk_infos, results):
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "image_id": image_info["id"],
                    "category_id": cls + 1,  # YOLO: 0-индексирован
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                })

    return detections


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------

def compute_map(coco_gt: COCO, detections: list[dict]) -> tuple[float, float]:
    if not detections:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])
    return map_50, map_50_95


def bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    union = aw * ah + bw * bh - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_precision_recall_f1(
    coco_gt: COCO, detections: list[dict], conf_threshold: float, iou_threshold: float
) -> tuple[dict, dict]:
    """
    TP/FP/FN по фиксированным порогам уверенности и IoU, усреднение по классам.
    Детекции внутри каждого класса разбираются по убыванию score — так более
    уверенное предсказание "забирает" совпавший GT-бокс первым.
    """
    gt_by_key: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        gt_by_key[(ann["image_id"], ann["category_id"])].append(ann["bbox"])

    matched = {key: [False] * len(boxes) for key, boxes in gt_by_key.items()}

    dets = [det for det in detections if det["score"] >= conf_threshold]
    dets.sort(key=lambda det: -det["score"])

    tp = defaultdict(int)
    fp = defaultdict(int)

    for det in dets:
        key = (det["image_id"], det["category_id"])
        gt_boxes = gt_by_key.get(key, [])

        best_iou, best_idx = 0.0, -1
        for idx, gt_box in enumerate(gt_boxes):
            if matched[key][idx]:
                continue
            iou = bbox_iou_xywh(det["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_iou >= iou_threshold:
            matched[key][best_idx] = True
            tp[det["category_id"]] += 1
        else:
            fp[det["category_id"]] += 1

    fn = defaultdict(int)
    for (_, category_id), flags in matched.items():
        fn[category_id] += flags.count(False)

    per_class = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        category_id = idx + 1
        t, f, n = tp[category_id], fp[category_id], fn[category_id]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1}

    overall = {
        metric: sum(c[metric] for c in per_class.values()) / len(per_class)
        for metric in ("precision", "recall", "f1")
    }

    return overall, per_class


# --------------------------------------------------------------------------

def resolve_checkpoint(model_name: str, checkpoints_dir: Path, tag: str) -> Path:
    if model_name == "detr":
        return checkpoints_dir / f"detr_{tag}"
    if model_name == "yolo":
        return checkpoints_dir / f"yolo_{tag}.pt"
    return checkpoints_dir / f"{model_name}_{tag}.pth"


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)

    coco_gt = COCO(args.val_ann)

    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "logs" / "metrics_summary.json"
    per_class_path = results_dir / "logs" / "metrics_per_class.json"

    # подгружаем то, что уже посчитано в предыдущих запусках (каждый запуск —
    # одна модель, свой процесс), и только дополняем/обновляем запись текущей
    summary_by_model = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            summary_by_model = {entry["model"]: entry for entry in json.load(file)}

    per_class_all = {}
    if per_class_path.exists():
        with open(per_class_path, encoding="utf-8") as file:
            per_class_all = json.load(file)

    for model_name in args.models:
        checkpoint_path = resolve_checkpoint(model_name, checkpoints_dir, args.checkpoint_tag)
        if not checkpoint_path.exists():
            print(f"[{model_name}] чекпоинт не найден: {checkpoint_path} — пропуск")
            continue

        print(f"=== {model_name} ({checkpoint_path.name}) ===")

        if model_name in ("faster_rcnn", "ssd", "retinanet"):
            detections = detect_torchvision(model_name, checkpoint_path, args.images_dir, args.val_ann, device)
        elif model_name == "detr":
            detections = detect_detr(checkpoint_path, args.images_dir, args.val_ann, device)
        else:  # yolo
            detections = detect_yolo(checkpoint_path, args.images_dir, coco_gt, str(device))

        map50, map50_95 = compute_map(coco_gt, detections)
        overall, per_class = compute_precision_recall_f1(
            coco_gt, detections, args.conf_threshold, args.iou_threshold
        )

        summary_by_model[model_name] = {
            "model": model_name,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
        }
        per_class_all[model_name] = per_class

        print(
            f"mAP50={map50:.4f} mAP50-95={map50_95:.4f} "
            f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} f1={overall['f1']:.4f}"
        )

        # сохраняем после каждой модели, а не только в конце — если следующая
        # модель упадёт, результаты по уже посчитанным моделям не потеряются
        summary = [summary_by_model[name] for name in ALL_MODELS if name in summary_by_model]
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        with open(per_class_path, "w", encoding="utf-8") as file:
            json.dump(per_class_all, file, indent=2, ensure_ascii=False)

    print(f"Готово. Метрики сохранены в {summary_path} и {per_class_path}")


if __name__ == "__main__":
    main()
