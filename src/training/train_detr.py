"""
Обучение DETR. Отдельный скрипт от train.py, т.к. у DETR (HuggingFace)
другой API работы с батчами и лоссом, но результаты сохраняются
в тот же results/ и в том же формате истории — для единого сравнения моделей.

Использование:
    python -m src.training.train_detr --epochs 10
"""
import argparse
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.detr_dataset import DetrCocoDataset, DetrCollateFn
from src.dataset.classes import CLASS_NAMES
from src.models.detr_factory import get_detr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
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


def move_encoding_to_device(encoding, device):
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask = encoding["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in encoding["labels"]]
    return pixel_values, pixel_mask, labels


def train_one_epoch(model, loader, optimizer, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_skipped = 0
    use_amp = device.type == "cuda"

    for encoding in loader:
        pixel_values, pixel_mask, labels = move_encoding_to_device(encoding, device)

        optimizer.zero_grad()

        try:
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss
        except ValueError as error:
            # matcher/generalized_box_iou считается внутри autocast вместе с
            # остальным forward-проходом; при fp16 предсказанные боксы иногда
            # округляются так, что x2/y2 чуть меньше x1/y0 — matcher падает
            # с ValueError вместо nan-лосса. Пропускаем такой батч, как и nan
            if "(corner) format" not in str(error):
                raise
            n_skipped += 1
            continue

        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_skipped > 0:
        print(f"  Пропущено батчей из-за nan/inf loss: {n_skipped}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for encoding in loader:
        pixel_values, pixel_mask, labels = move_encoding_to_device(encoding, device)
        try:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        except ValueError as error:
            # см. комментарий в train_one_epoch — та же fp16/расходящаяся-модель
            # причина может проявиться и на валидации
            if "(corner) format" not in str(error):
                raise
            continue

        if torch.isfinite(outputs.loss):
            total_loss += outputs.loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    model, processor = get_detr(num_classes=len(CLASS_NAMES))
    model.to(device)

    collate_fn = DetrCollateFn(processor)

    train_dataset = DetrCocoDataset(args.images_dir, args.train_ann)
    val_dataset = DetrCocoDataset(args.images_dir, args.val_ann)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"),
    )

    # backbone (предобученный ResNet50) дообучаем медленнее, чем "новые" слои
    # (transformer-голову) — стандартная практика для DETR
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    history = {"train_loss": [], "val_loss": []}
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
            f"[detr] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(results_dir / "checkpoints" / "detr_best")
            processor.save_pretrained(results_dir / "checkpoints" / "detr_best")
            print(f"  Новый лучший val_loss={val_loss:.4f} — сохранены веса detr_best/")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"  val_loss не улучшается {args.patience} эпох подряд — "
                    f"ранняя остановка на эпохе {epoch}"
                )
                break

    # веса последней эпохи — для отладки/сравнения с лучшей эпохой; для итоговых
    # результатов и сравнения моделей использовать checkpoints/detr_best/
    model.save_pretrained(results_dir / "checkpoints" / "detr_last")
    processor.save_pretrained(results_dir / "checkpoints" / "detr_last")

    with open(results_dir / "logs" / "detr_history.json", "w") as file:
        json.dump(history, file, indent=2)

    print(f"Готово. Веса и логи сохранены в {results_dir}/")


if __name__ == "__main__":
    main()