"""
Экспорт датасета KITTI в формат YOLO (Ultralytics).

Создаёт структуру:
    data/processed/yolo/images/train/*.png
    data/processed/yolo/images/val/*.png
    data/processed/yolo/labels/train/*.txt
    data/processed/yolo/labels/val/*.txt
    data/processed/yolo/dataset.yaml

Использует тот же train.txt/val.txt split и тот же kitti_loader.py,
что и coco_export.py — единая точка правды по разбиению данных
и по тому, какие классы считаются валидными (IGNORED_CLASSES).

Использование:
    python -m src.dataset.yolo_export
"""
from pathlib import Path
import shutil
import yaml

from src.dataset.classes import CLASSES
from src.dataset.datatypes import AnnotatedImage, BoundingBox
from src.dataset.kitti_loader import parse_dataset


def load_split_ids(split_path: Path) -> set[str]:
    """
    Считывает список image_id из train.txt / val.txt.
    Та же функция, что в coco_export.py — оставлена здесь копией,
    чтобы модуль не зависел от другого экспортера.
    """
    with open(split_path, "r") as file:
        return {line.strip() for line in file if line.strip()}


def to_yolo_line(box: BoundingBox, img_width: int, img_height: int) -> str:
    """
    Конвертирует BoundingBox (x1, y1, x2, y2 в пикселях) в строку YOLO:
    class_id x_center y_center width height — все координаты нормализованы в [0, 1].
    """
    x_center = (box.x1 + box.x2) / 2 / img_width
    y_center = (box.y1 + box.y2) / 2 / img_height
    width = box.width / img_width
    height = box.height / img_height

    class_id = CLASSES[box.class_name]

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def export_split(
    dataset: list[AnnotatedImage],
    split_ids: set[str],
    out_images_dir: Path,
    out_labels_dir: Path,
) -> int:
    """
    Копирует изображения и пишет .txt-аннотации для одного split'а.
    Возвращает число экспортированных изображений.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    subset = [img for img in dataset if img.image_id in split_ids]

    if not subset:
        raise ValueError(
            f"Ни одно изображение не попало в split. "
            f"Проверь, что id в split-файле совпадают с именами файлов."
        )

    for annotated_image in subset:
        # копируем (не симлинк) — устойчивее на Windows без прав администратора
        dst_image = out_images_dir / annotated_image.image_path.name
        if not dst_image.exists():
            shutil.copy(annotated_image.image_path, dst_image)

        lines = [
            to_yolo_line(box, annotated_image.width, annotated_image.height)
            for box in annotated_image.boxes
        ]

        # пустой файл — валидный случай в YOLO (изображение без объектов)
        label_path = out_labels_dir / f"{annotated_image.image_id}.txt"
        with open(label_path, "w") as file:
            file.write("\n".join(lines))

    return len(subset)


def write_dataset_yaml(output_path: Path, dataset_root: Path) -> None:
    """
    Пишет dataset.yaml, который ожидает ultralytics.
    """
    config = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for name, idx in CLASSES.items()},
    }

    with open(output_path, "w") as file:
        yaml.dump(config, file, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    IMAGES_DIR = Path("data/raw/kitti/training/image_2")
    LABELS_DIR = Path("data/raw/kitti/training/label_2")
    SPLITS_DIR = Path("data/processed/splits")
    OUT_DIR = Path("data/processed/yolo")

    full_dataset = parse_dataset(IMAGES_DIR, LABELS_DIR)

    train_ids = load_split_ids(SPLITS_DIR / "train.txt")
    val_ids = load_split_ids(SPLITS_DIR / "val.txt")

    n_train = export_split(
        full_dataset, train_ids, OUT_DIR / "images" / "train", OUT_DIR / "labels" / "train"
    )
    n_val = export_split(
        full_dataset, val_ids, OUT_DIR / "images" / "val", OUT_DIR / "labels" / "val"
    )

    write_dataset_yaml(OUT_DIR / "dataset.yaml", OUT_DIR)

    print(f"train: {n_train} изображений, val: {n_val} изображений")
    print(f"dataset.yaml сохранён в {OUT_DIR / 'dataset.yaml'}")