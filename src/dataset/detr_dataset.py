"""
Dataset и collate_fn для обучения DETR (HuggingFace transformers).

DETR устроен иначе, чем модели torchvision.models.detection:
    - вход в модель: pixel_values + pixel_mask (не просто список тензоров)
    - target: class_labels (0-индексированные, БЕЗ фона) + boxes в
      нормализованном формате cxcywh [0, 1]
    - вся эта конвертация инкапсулирована в DetrImageProcessor, поэтому
      Dataset здесь отдаёт "сырые" PIL-изображения + аннотации в родном
      COCO-формате, а не готовые тензоры

Переиспользует train.json/val.json из coco_export.py — отдельная подготовка
данных для DETR не нужна, только пересчёт category_id (там он +1 из-за
торчвижн-моделей, здесь нужен 0-индексированный).
"""
from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset


class DetrCocoDataset(Dataset):
    """
    Параметры
    ----------
    images_dir : Path
        Папка с изображениями.
    annotations_path : Path
        train.json или val.json, созданные coco_export.py.
    """

    def __init__(self, images_dir: Path, annotations_path: Path):
        self.images_dir = Path(images_dir)

        with open(annotations_path, "r") as file:
            coco = json.load(file)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        self.annotations_by_image: dict[int, list[dict]] = {
            image_id: [] for image_id in self.image_ids
        }
        for ann in coco["annotations"]:
            # -1: в coco_export.py category_id сдвинут на +1 под torchvision (фон=0),
            # а DetrForObjectDetection ожидает 0-индексированные классы без фона
            shifted_ann = dict(ann)
            shifted_ann["category_id"] = ann["category_id"] - 1
            self.annotations_by_image[ann["image_id"]].append(shifted_ann)

        # categories тоже пересчитываем для консистентности (не используется
        # напрямую процессором, но пригодится при отладке/визуализации)
        self.categories = {
            cat["id"] - 1: cat["name"] for cat in coco["categories"]
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_info = self.images[image_id]

        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        coco_annotation = {
            "image_id": image_id,
            "annotations": self.annotations_by_image[image_id],
        }

        return image, coco_annotation


class DetrCollateFn:
    """
    Класс вместо вложенной функции: на Windows DataLoader с num_workers > 0
    использует multiprocessing.spawn, который требует, чтобы collate_fn была
    picklable — вложенные функции (замыкания) таковыми не являются, классы — да.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, annotations = zip(*batch)

        encoding = self.processor(
            images=list(images),
            annotations=list(annotations),
            return_tensors="pt",
        )

        return encoding