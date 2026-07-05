"""
Обучение YOLO11n на экспортированном датасете (data/processed/yolo/dataset.yaml).

По интерфейсу и месту сохранения результатов согласован с train.py и train_detr.py,
хотя сам ultralytics использует свой внутренний способ обучения/логирования.

Использование:
    python -m src.training.train_yolo --epochs 10
"""
import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dataset-yaml", default="data/processed/yolo/dataset.yaml")
    parser.add_argument("--checkpoint", default="yolo11n.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Остановить обучение, если метрики не улучшаются N эпох подряд",
    )
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.checkpoint)

    results_dir = Path(args.results_dir)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_results = model.train(
        data=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        seed=args.seed,
        patience=args.patience,
        project=str(results_dir / "yolo_runs"),
        name="yolo11n",
        exist_ok=True,
    )

    # ultralytics сам пишет веса/метрики в свою внутреннюю структуру папок —
    # копируем финальный результат в общий results/, как у остальных моделей,
    # чтобы сравнение моделей шло из одного места. best.pt ultralytics выбирает
    # сам по fitness-метрике на каждой эпохе (аналог best_val_loss у других моделей).
    run_dir = Path(train_results.save_dir)

    best_weights = run_dir / "weights" / "best.pt"
    shutil.copy(best_weights, results_dir / "checkpoints" / "yolo_best.pt")

    last_weights = run_dir / "weights" / "last.pt"
    shutil.copy(last_weights, results_dir / "checkpoints" / "yolo_last.pt")

    metrics_csv = run_dir / "results.csv"
    shutil.copy(metrics_csv, results_dir / "logs" / "yolo_results.csv")

    print(f"Готово. Веса и логи сохранены в {results_dir}/")


if __name__ == "__main__":
    main()