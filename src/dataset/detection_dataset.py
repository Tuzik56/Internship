"""
Единый Dataset для моделей из torchvision.models.detection
(Faster R-CNN, SSD, RetinaNet), а также применим для DETR
(с отдельным препроцессингом от HuggingFace поверх тех же данных).

Читает COCO JSON, созданный src/dataset/coco_export.py.
"""
from pathlib import Path
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CocoDetectionDataset(Dataset):
    """
    Датасет для torchvision-моделей детекции.

    Каждый элемент — (image, target), где:
        image: FloatTensor [3, H, W], нормализован в [0, 1]
        target: dict с ключами:
            boxes: FloatTensor [N, 4] в формате xyxy
            labels: LongTensor [N]
            image_id: LongTensor [1]
            area: FloatTensor [N]
            iscrowd: LongTensor [N]
    """

    def __init__(self, images_dir: Path, annotations_path: Path):
        self.images_dir = Path(images_dir)

        with open(annotations_path, "r") as file:
            coco = json.load(file)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        # группируем аннотации по image_id, чтобы не делать O(n^2) поиск
        self.annotations_by_image: dict[int, list[dict]] = {
            image_id: [] for image_id in self.image_ids
        }
        for annotation in coco["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_info = self.images[image_id]

        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)  # [0, 1], [3, H, W]

        anns = self.annotations_by_image[image_id]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        img_width = image_info["width"]
        img_height = image_info["height"]

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # обрезаем по границам изображения (в KITTI встречаются боксы,
            # выходящие за край кадра у усечённых объектов)
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(img_width), x + w)
            y2 = min(float(img_height), y + h)

            # пропускаем вырожденные боксы (нулевая/отрицательная ширина или высота) —
            # именно они чаще всего вызывают nan в loss у SSD
            if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(ann["iscrowd"])

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # изображение без объектов (в KITTI встречается редко, но нужно обработать)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id]),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return image_tensor, target


def collate_fn(batch):
    """
    У каждого изображения разное число объектов, поэтому нельзя
    просто сложить target'ы в тензор — используем список кортежей.
    """
    return tuple(zip(*batch))