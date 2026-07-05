"""
Качественное демо на видео (KITTI Tracking): прогон обученных детекторов
по кадрам одной последовательности, отрисовка предсказанных боксов,
сохранение в .mp4 — раздел 8.6 "визуализация предсказаний".

Track ID и ground truth не используются: отслеживание объектов не требуется
по заданию, кадры обрабатываются независимо, как обычные изображения.

Использование:
    python -m src.evaluation.video_demo --sequence 0000
    python -m src.evaluation.video_demo --sequence 0000 --models yolo faster_rcnn
"""
import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tv_f
from PIL import Image

from src.dataset.classes import CLASS_NAMES
from src.models.model_factory import get_model

ALL_MODELS = ["faster_rcnn", "ssd", "retinanet", "detr", "yolo"]

CLASS_COLORS_BGR = {
    "Car": (0, 200, 0),
    "Van": (0, 140, 255),
    "Truck": (255, 140, 0),
    "Pedestrian": (0, 0, 255),
    "Cyclist": (255, 0, 255),
    "Misc": (160, 160, 160),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="0000")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--checkpoint-tag", default="best", choices=["best", "last"])
    parser.add_argument("--tracking-dir", default="data/raw/kitti_tracking/training")
    parser.add_argument("--checkpoints-dir", default="results/checkpoints")
    parser.add_argument("--out-dir", default="results/video_demo")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None, help="Ограничить число кадров (для быстрой проверки)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_checkpoint(model_name: str, checkpoints_dir: Path, tag: str) -> Path:
    if model_name == "detr":
        return checkpoints_dir / f"detr_{tag}"
    if model_name == "yolo":
        return checkpoints_dir / f"yolo_{tag}.pt"
    return checkpoints_dir / f"{model_name}_{tag}.pth"


# --------------------------------------------------------------------------
# Детекторы: единый интерфейс detect(image: PIL.Image) -> list[(class_name, score, (x1,y1,x2,y2))]
# --------------------------------------------------------------------------

def load_torchvision_detector(model_name: str, checkpoint_path: Path, device, conf_threshold: float):
    model = get_model(model_name, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    @torch.no_grad()
    def detect(image: Image.Image):
        tensor = tv_f.to_tensor(image).to(device)
        output = model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        predictions = []
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue
            class_name = CLASS_NAMES[label - 1]  # torchvision: 0 зарезервирован под фон
            predictions.append((class_name, float(score), tuple(box.tolist())))
        return predictions

    return detect


def load_detr_detector(checkpoint_dir: Path, device, conf_threshold: float):
    from transformers import DetrForObjectDetection, DetrImageProcessor

    processor = DetrImageProcessor.from_pretrained(checkpoint_dir)
    model = DetrForObjectDetection.from_pretrained(checkpoint_dir).to(device).eval()

    @torch.no_grad()
    def detect(image: Image.Image):
        encoding = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**encoding)

        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        result = processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_sizes
        )[0]

        predictions = []
        for box, score, label in zip(
            result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()
        ):
            predictions.append((CLASS_NAMES[label], float(score), tuple(box)))
        return predictions

    return detect


def load_yolo_detector(checkpoint_path: Path, device, conf_threshold: float):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))

    def detect(image: Image.Image):
        result = model.predict(image, conf=conf_threshold, device=str(device), verbose=False)[0]

        predictions = []
        for box in result.boxes:
            class_name = CLASS_NAMES[int(box.cls[0])]
            score = float(box.conf[0])
            xyxy = tuple(box.xyxy[0].tolist())
            predictions.append((class_name, score, xyxy))
        return predictions

    return detect


def build_detector(model_name: str, checkpoint_path: Path, device, conf_threshold: float):
    if model_name in ("faster_rcnn", "ssd", "retinanet"):
        return load_torchvision_detector(model_name, checkpoint_path, device, conf_threshold)
    if model_name == "detr":
        return load_detr_detector(checkpoint_path, device, conf_threshold)
    return load_yolo_detector(checkpoint_path, device, conf_threshold)


# --------------------------------------------------------------------------

def draw_predictions(frame_bgr: np.ndarray, predictions: list, model_name: str) -> np.ndarray:
    frame = frame_bgr.copy()

    for class_name, score, (x1, y1, x2, y2) in predictions:
        color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)

        label = f"{class_name} {score:.2f}"
        cv2.putText(frame, label, (p1[0], max(p1[1] - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(frame, model_name, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def render_sequence(model_name: str, detect, frame_paths: list, out_path: Path, fps: int) -> None:
    with Image.open(frame_paths[0]) as first_frame:
        width, height = first_frame.size

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB")
        predictions = detect(image)

        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_bgr = draw_predictions(frame_bgr, predictions, model_name)
        writer.write(frame_bgr)

        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            print(f"  {i + 1}/{len(frame_paths)} кадров")

    writer.release()


def main():
    args = parse_args()
    device = torch.device(args.device)

    frames_dir = Path(args.tracking_dir) / "image_02" / args.sequence
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"Кадры не найдены: {frames_dir}")
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    checkpoints_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        checkpoint_path = resolve_checkpoint(model_name, checkpoints_dir, args.checkpoint_tag)
        if not checkpoint_path.exists():
            print(f"[{model_name}] чекпоинт не найден: {checkpoint_path} — пропуск")
            continue

        print(f"=== {model_name} ({len(frame_paths)} кадров, seq {args.sequence}) ===")
        detect = build_detector(model_name, checkpoint_path, device, args.conf_threshold)

        out_path = out_dir / f"{model_name}_seq{args.sequence}.mp4"
        render_sequence(model_name, detect, frame_paths, out_path, args.fps)
        print(f"  Сохранено: {out_path}")

        del detect
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Готово. Видео сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
