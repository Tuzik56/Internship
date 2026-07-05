from pathlib import Path
from PIL import Image
from src.dataset.datatypes import BoundingBox, AnnotatedImage
from src.dataset.classes import IGNORED_CLASSES


def parse_annotation(label_path: Path) -> list[BoundingBox]:
    """
    Считывает один файл аннотаций KITTI.

    Parameters
    ----------
    label_path : Path
        Путь к txt-файлу.

    Returns
    -------
    list[BoundingBox]
        Список объектов изображения.
    """

    boxes = []

    with open(label_path, "r") as file:

        for line in file:

            if not line.strip():
                continue

            parts = line.split()

            class_name = parts[0]

            if class_name in IGNORED_CLASSES:
                continue

            box = BoundingBox(
                class_name=class_name,
                x1=float(parts[4]),
                y1=float(parts[5]),
                x2=float(parts[6]),
                y2=float(parts[7]),
            )

            boxes.append(box)

    return boxes


def parse_dataset(images_dir: Path, labels_dir: Path) -> list[AnnotatedImage]:
    """
    Считывает весь датасет KITTI.

    Parameters
    ----------
    images_dir : Path
        Папка с изображениями.

    labels_dir : Path
        Папка с аннотациями.

    Returns
    -------
    list[AnnotatedImage]
    """

    dataset = []

    image_paths = sorted(images_dir.glob("*.png"))

    for image_path in image_paths:

        label_path = labels_dir / f"{image_path.stem}.txt"

        boxes = parse_annotation(label_path)

        with Image.open(image_path) as image:

            width, height = image.size

        annotated_image = AnnotatedImage(
            image_path=image_path,
            width=width,
            height=height,
            boxes=boxes,
        )

        dataset.append(annotated_image)

    return dataset


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    images_dir = (
        project_root
        / "data"
        / "raw"
        / "kitti"
        / "training"
        / "image_2"
    )

    labels_dir = (
        project_root
        / "data"
        / "raw"
        / "kitti"
        / "training"
        / "label_2"
    )

    dataset = parse_dataset(images_dir, labels_dir)

    print(f"Изображений: {len(dataset)}\n")

    print(dataset[0])
    print(dataset[1])
    print(dataset[2])