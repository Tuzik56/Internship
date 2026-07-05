"""
Таблицы и графики для сравнения моделей по результатам, уже посчитанным
src/evaluation/metrics.py (results/logs/metrics_summary.json,
metrics_per_class.json) и обучением (results/logs/{model}_history.json,
yolo_results.csv).

Использование:
    python -m src.evaluation.plots
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_ORDER = ["faster_rcnn", "ssd", "retinanet", "detr", "yolo"]
METRIC_KEYS = ["mAP50", "mAP50-95", "precision", "recall", "f1"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def load_summary(results_dir: Path) -> list[dict]:
    with open(results_dir / "logs" / "metrics_summary.json", encoding="utf-8") as file:
        summary = json.load(file)

    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return sorted(summary, key=lambda row: order.get(row["model"], len(order)))


def print_summary_table(summary: list[dict]) -> None:
    header = ["model"] + METRIC_KEYS
    print(" | ".join(f"{col:>12}" for col in header))
    for row in summary:
        values = [row["model"]] + [f"{row[metric]:.4f}" for metric in METRIC_KEYS]
        print(" | ".join(f"{value:>12}" for value in values))


def save_summary_csv(summary: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model"] + METRIC_KEYS)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in ["model"] + METRIC_KEYS})


def save_per_class_csv(per_class_all: dict, out_path: Path) -> None:
    class_names = list(next(iter(per_class_all.values())).keys())

    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "class", "precision", "recall", "f1"])
        for model_name in MODEL_ORDER:
            if model_name not in per_class_all:
                continue
            for class_name in class_names:
                row = per_class_all[model_name][class_name]
                writer.writerow([model_name, class_name, row["precision"], row["recall"], row["f1"]])


def plot_metrics_comparison(summary: list[dict], out_path: Path) -> None:
    """
    Групповая столбчатая диаграмма: mAP50/mAP50-95/Precision/Recall/F1 по моделям.
    В отличие от loss, эти метрики считаются одинаково для всех моделей
    (см. src/evaluation/metrics.py) — их можно сравнивать напрямую.
    """
    models = [row["model"] for row in summary]
    x = range(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(METRIC_KEYS):
        values = [row[metric] for row in summary]
        offsets = [xi + (i - len(METRIC_KEYS) / 2) * width + width / 2 for xi in x]
        ax.bar(offsets, values, width=width, label=metric)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение моделей по метрикам качества")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_loss_history(model_name: str, results_dir: Path) -> tuple[list[float], list[float]] | None:
    """Возвращает (train_loss, val_loss) по эпохам, либо None, если лог не найден."""
    if model_name == "yolo":
        csv_path = results_dir / "logs" / "yolo_results.csv"
        if not csv_path.exists():
            return None

        train_loss, val_loss = [], []
        with open(csv_path, newline="") as file:
            for row in csv.DictReader(file):
                # box+cls+dfl — суммарный loss YOLO за эпоху, чтобы получить
                # одно число, сопоставимое по смыслу (не по шкале!) с
                # train_loss/val_loss остальных моделей
                train_loss.append(
                    float(row["train/box_loss"]) + float(row["train/cls_loss"]) + float(row["train/dfl_loss"])
                )
                val_loss.append(
                    float(row["val/box_loss"]) + float(row["val/cls_loss"]) + float(row["val/dfl_loss"])
                )
        return train_loss, val_loss

    history_path = results_dir / "logs" / f"{model_name}_history.json"
    if not history_path.exists():
        return None

    with open(history_path, encoding="utf-8") as file:
        history = json.load(file)
    return history["train_loss"], history["val_loss"]


def plot_loss_curves(results_dir: Path, out_path: Path) -> None:
    """
    Loss каждой модели — на своей оси (не всех вместе): у Faster R-CNN/SSD/
    RetinaNet/DETR/YOLO разные компоненты и масштабы loss, совмещать их на
    одном графике вводит в заблуждение (сравнима форма кривой, не значения).
    """
    available = [
        (model_name, load_loss_history(model_name, results_dir))
        for model_name in MODEL_ORDER
    ]
    available = [(name, history) for name, history in available if history is not None]

    if not available:
        print("Нет ни одного файла с историей обучения — график loss не построен")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)

    for ax, (model_name, (train_loss, val_loss)) in zip(axes[0], available):
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="train_loss")
        ax.plot(epochs, val_loss, label="val_loss")
        ax.set_title(model_name)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("Кривые обучения по моделям (шкалы loss не сравнимы между моделями)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class_f1(per_class_all: dict, out_path: Path) -> None:
    class_names = list(next(iter(per_class_all.values())).keys())
    models = [name for name in MODEL_ORDER if name in per_class_all]

    x = range(len(class_names))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(models):
        values = [per_class_all[model_name][cls]["f1"] for cls in class_names]
        offsets = [xi + (i - len(models) / 2) * width + width / 2 for xi in x]
        ax.bar(offsets, values, width=width, label=model_name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("F1 по классам и моделям")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_hp_search_metric(run_dir: Path, model_name: str, metric: str = "mAP50") -> float | None:
    summary_path = run_dir / "logs" / "metrics_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)

    for row in summary:
        if row["model"] == model_name:
            return row[metric]
    return None


def plot_hp_search_ssd_lr(hp_search_dir: Path, out_path: Path) -> None:
    """mAP50 в зависимости от learning rate — эксперимент из раздела "Эксперименты"."""
    lr_values = [0.0005, 0.001, 0.005]
    points = [
        (lr, load_hp_search_metric(hp_search_dir / f"ssd_lr_{lr}", "ssd"))
        for lr in lr_values
    ]
    points = [(lr, value) for lr, value in points if value is not None]
    if not points:
        print("Нет данных hp_search для SSD — график lr пропущен")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([lr for lr, _ in points], [value for _, value in points], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (лог. шкала)")
    ax.set_ylabel("mAP50")
    ax.set_title("SSD: качество в зависимости от learning rate (15 эпох)")
    for lr, value in points:
        ax.annotate(f"{value:.3f}", (lr, value), textcoords="offset points", xytext=(0, 8), ha="center")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hp_search_detr_backbone_lr(hp_search_dir: Path, out_path: Path) -> None:
    """mAP50 в зависимости от lr backbone — эксперимент из раздела "Эксперименты"."""
    variants = [("1e-5\n(дифференцированный)", "detr_backbone_1e-5"), ("1e-4\n(равный lr головы)", "detr_backbone_1e-4")]
    points = [
        (label, load_hp_search_metric(hp_search_dir / dirname, "detr"))
        for label, dirname in variants
    ]
    points = [(label, value) for label, value in points if value is not None]
    if not points:
        print("Нет данных hp_search для DETR — график lr backbone пропущен")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([label for label, _ in points], [value for _, value in points])
    ax.set_ylabel("mAP50")
    ax.set_title("DETR: качество в зависимости от lr backbone (10 эпох)")
    for bar, (_, value) in zip(bars, points):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 5), ha="center")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(results_dir)
    print_summary_table(summary)
    save_summary_csv(summary, results_dir / "logs" / "metrics_summary.csv")

    plot_metrics_comparison(summary, plots_dir / "metrics_comparison.png")
    plot_loss_curves(results_dir, plots_dir / "loss_curves.png")

    per_class_path = results_dir / "logs" / "metrics_per_class.json"
    if per_class_path.exists():
        with open(per_class_path, encoding="utf-8") as file:
            per_class_all = json.load(file)
        save_per_class_csv(per_class_all, results_dir / "logs" / "metrics_per_class.csv")
        plot_per_class_f1(per_class_all, plots_dir / "per_class_f1.png")

    hp_search_dir = results_dir / "hp_search"
    if hp_search_dir.exists():
        plot_hp_search_ssd_lr(hp_search_dir, plots_dir / "hp_search_ssd_lr.png")
        plot_hp_search_detr_backbone_lr(hp_search_dir, plots_dir / "hp_search_detr_backbone_lr.png")

    print(f"Готово. Таблицы и графики сохранены в {results_dir}/logs/ и {plots_dir}/")


if __name__ == "__main__":
    main()
