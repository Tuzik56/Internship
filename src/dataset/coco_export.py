"""
Экспорт датасета KITTI в формат COCO.

Нужен для унифицированного обучения моделей, использующих
torchvision.models.detection (Faster R-CNN, SSD, RetinaNet)
и DETR (HuggingFace), поверх уже существующего kitti_loader.py.

Использование:
    python -m src.dataset.coco_export
"""
from pathlib import Path
import json

from src.dataset.classes import CLASSES
from src.dataset.datatypes import AnnotatedImage
from src.dataset.kitti_loader import parse_dataset


def load_split_ids(split_path: Path) -> set[str]:
    """
    Считывает список image_id (номеров файлов) из train.txt / val.txt.
    """
    with open(split_path, "r") as file:
        return {line.strip() for line in file if line.strip()}


def build_coco_dict(dataset: list[AnnotatedImage]) -> dict:
    """
    Преобразует список AnnotatedImage в словарь формата COCO.

    Важно: category_id смещён на +1 относительно CLASSES,
    так как torchvision.models.detection резервирует id=0 под фон.
    """
    images = []
    annotations = []
    annotation_id = 1

    for annotated_image in dataset:
        # COCO требует числовой id изображения; используем позицию как int,
        # а исходное имя файла храним отдельно для отладки/визуализации
        numeric_id = int(annotated_image.image_id)

        images.append({
            "id": numeric_id,
            "file_name": annotated_image.image_path.name,
            "width": annotated_image.width,
            "height": annotated_image.height,
        })

        for box in annotated_image.boxes:
            annotations.append({
                "id": annotation_id,
                "image_id": numeric_id,
                "category_id": CLASSES[box.class_name] + 1,  # +1: 0 зарезервирован под фон
                "bbox": [box.x1, box.y1, box.width, box.height],  # COCO формат: x, y, w, h
                "area": box.area,
                "iscrowd": 0,
            })
            annotation_id += 1

    categories = [
        {"id": idx + 1, "name": name}
        for name, idx in CLASSES.items()
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def export_split(
    images_dir: Path,
    labels_dir: Path,
    split_path: Path,
    output_path: Path,
) -> None:
    """
    Строит COCO JSON для одного split'а (train или val) и сохраняет на диск.
    """
    split_ids = load_split_ids(split_path)

    full_dataset = parse_dataset(images_dir, labels_dir)
    subset = [img for img in full_dataset if img.image_id in split_ids]

    if not subset:
        raise ValueError(
            f"Ни одно изображение не попало в {split_path.name}. "
            f"Проверь, что номера в split-файле совпадают с именами файлов."
        )

    coco_dict = build_coco_dict(subset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(coco_dict, file, indent=2)

    n_images = len(coco_dict["images"])
    n_annotations = len(coco_dict["annotations"])
    print(f"{output_path.name}: {n_images} изображений, {n_annotations} аннотаций")


if __name__ == "__main__":
    # Определяем корень проекта относительно этого файла (поднимаемся на 3 уровня вверх из src/dataset)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Теперь все пути строятся строго от корня проекта, где бы вы ни находились
    IMAGES_DIR = PROJECT_ROOT / "data/raw/kitti/training/image_2"
    LABELS_DIR = PROJECT_ROOT / "data/raw/kitti/training/label_2"
    SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
    OUT_DIR = PROJECT_ROOT / "data/processed/annotations"

    export_split(IMAGES_DIR, LABELS_DIR, SPLITS_DIR / "train.txt", OUT_DIR / "train.json")
    export_split(IMAGES_DIR, LABELS_DIR, SPLITS_DIR / "val.txt", OUT_DIR / "val.json")
