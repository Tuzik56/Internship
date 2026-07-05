from pathlib import Path
from collections import Counter
from PIL import Image
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ----------------------------
# Пути к датасету
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "kitti" / "training" / "image_2"
LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "kitti" / "training" / "label_2"


def count_images_and_labels():
    """
    Подсчитывает количество изображений и файлов разметки.
    """

    image_files = sorted(IMAGES_DIR.glob("*.png"))
    label_files = sorted(LABELS_DIR.glob("*.txt"))

    print(f"Изображений : {len(image_files)}")
    print(f"Аннотаций   : {len(label_files)}")


def get_class_statistics():
    """Находит все классы объектов в датасете и считает их количество."""

    # Используем Counter вместо set
    classes_counter = Counter()

    for label_file in LABELS_DIR.glob("*.txt"):
        with open(label_file, "r") as f:

            for line in f:
                # Проверяем, что строка не пустая, чтобы избежать ошибок
                if line.strip():
                    object_class = line.split()[0]
                    # Увеличиваем счетчик для этого класса
                    classes_counter[object_class] += 1

    print("\nКлассы:")

    # Сортируем по имени класса и выводим
    for cls, count in sorted(classes_counter.items()):
        print(f"{cls} {count}")


def analyze_image_sizes():
    """
    Анализирует размеры изображений.
    """

    size_counter = Counter()

    for image_file in IMAGES_DIR.glob("*.png"):

        with Image.open(image_file) as image:

            size_counter[image.size] += 1

    print("\nРазмеры изображений:")

    for size, count in sorted(size_counter.items()):
        print(f"{size[0]} x {size[1]} : {count}")


def show_example():
    """
    Показывает случайное изображение с разметкой KITTI.
    """

    image_files = sorted(IMAGES_DIR.glob("*.png"))

    image_path = random.choice(image_files)

    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(image)

    with open(label_path, "r") as f:

        for line in f:

            if not line.strip():
                continue

            parts = line.split()

            object_class = parts[0]

            if object_class == "DontCare":
                continue

            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                object_class,
                color="yellow",
                fontsize=10,
                backgroundcolor="black"
            )

    ax.set_title(image_path.name)
    ax.axis("off")

    plt.show()


if __name__ == "__main__":

    count_images_and_labels()

    get_class_statistics()

    analyze_image_sizes()

    show_example()