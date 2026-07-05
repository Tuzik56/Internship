"""
Единый скрипт обучения для моделей torchvision.models.detection.

Использование:
    python -m src.training.train --model faster_rcnn --epochs 10
    python -m src.training.train --model ssd --epochs 10
    python -m src.training.train --model retinanet --epochs 10

Гиперпараметры читаются из configs/<model>.yaml, если он есть,
иначе используются значения по умолчанию из аргументов командной строки.
"""
import argparse
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.detection_dataset import CocoDetectionDataset, collate_fn
from src.dataset.classes import CLASS_NAMES
from src.models.model_factory import get_model


DEFAULT_LR = {
    "faster_rcnn": 0.005,
    "ssd": 0.001,       # SSD чувствительнее к lr, при 0.005 склонен уходить в nan
    "retinanet": 0.005,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["faster_rcnn", "ssd", "retinanet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Остановить обучение, если val_loss не улучшается N эпох подряд",
    )
    parser.add_argument("--images-dir", default="data/raw/kitti/training/image_2")
    parser.add_argument("--train-ann", default="data/processed/annotations/train.json")
    parser.add_argument("--val-ann", default="data/processed/annotations/val.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_skipped = 0

    use_amp = device.type == "cuda"

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_skipped > 0:
        print(f"  Пропущено батчей из-за nan/inf loss: {n_skipped}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    """
    torchvision-модели в eval-режиме не отдают лосс (только предсказания),
    поэтому для val-лосса временно переключаем в train-режим,
    но без обновления весов (только forward, без backward/optimizer.step).
    Для итоговых метрик качества (mAP) используется отдельный скрипт evaluation/metrics.py.
    """
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.lr is None:
        args.lr = DEFAULT_LR[args.model]

    train_dataset = CocoDetectionDataset(args.images_dir, args.train_ann)
    val_dataset = CocoDetectionDataset(args.images_dir, args.val_ann)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"),
    )

    torch.backends.cudnn.benchmark = True  # ускоряет свёртки при фиксированном размере входа

    model = get_model(args.model, num_classes=len(CLASS_NAMES)).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    history = {"train_loss": [], "val_loss": []}
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    results_dir = Path(args.results_dir)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = evaluate_loss(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - start
        print(
            f"[{args.model}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), results_dir / "checkpoints" / f"{args.model}_best.pth")
            print(f"  Новый лучший val_loss={val_loss:.4f} — сохранены веса {args.model}_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"  val_loss не улучшается {args.patience} эпох подряд — "
                    f"ранняя остановка на эпохе {epoch}"
                )
                break

    # веса последней эпохи — для отладки/сравнения с лучшей эпохой; для итоговых
    # результатов и сравнения моделей использовать {model}_best.pth
    torch.save(model.state_dict(), results_dir / "checkpoints" / f"{args.model}_last.pth")

    with open(results_dir / "logs" / f"{args.model}_history.json", "w") as file:
        json.dump(history, file, indent=2)

    print(f"Готово. Веса и логи сохранены в {results_dir}/")


if __name__ == "__main__":
    main()