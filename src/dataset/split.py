from sklearn.model_selection import train_test_split
from src.dataset.kitti_loader import parse_dataset
from pathlib import Path


def split_dataset(
    dataset,
    test_size=0.2,
    random_state=42,
):
    """
    Разбивает датасет на обучающую и валидационную выборки.
    """

    train_dataset, val_dataset = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_dataset, val_dataset


def save_split_files(train_dataset, val_dataset, output_dir):
    """
    Сохраняет train.txt и val.txt.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"

    with open(train_file, "w") as f:

        for image in train_dataset:

            f.write(f"{image.image_id}\n")

    with open(val_file, "w") as f:

        for image in val_dataset:

            f.write(f"{image.image_id}\n")


def load_split_files(split_dir):
    """
    Загружает train.txt и val.txt.
    """

    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"

    with open(train_file) as f:
        train_ids = {line.strip() for line in f}

    with open(val_file) as f:
        val_ids = {line.strip() for line in f}

    return train_ids, val_ids


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

    train_dataset, val_dataset = split_dataset(dataset)

    print(len(train_dataset))
    print(len(val_dataset))

    split_dir = (
            project_root
            / "data"
            / "processed"
            / "splits"
    )

    save_split_files(train_dataset, val_dataset, split_dir)

    print("Разбиение сохранено.")

    train_ids, val_ids = load_split_files(split_dir)

    print(f"Train IDs: {len(train_ids)}")
    print(f"Validation IDs: {len(val_ids)}")

    intersection = train_ids & val_ids

    print(f"Общих изображений: {len(intersection)}")