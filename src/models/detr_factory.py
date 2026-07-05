"""
Загрузка предобученного DETR и его image processor.

Модель дообучается с 91 класса COCO на 6 классов KITTI —
голова классификации переинициализируется автоматически
благодаря ignore_mismatched_sizes=True.
"""
from transformers import DetrForObjectDetection, DetrImageProcessor

CHECKPOINT = "facebook/detr-resnet-50"

# то же разрешение, что у Faster R-CNN/RetinaNet (480-640) — для честного
# сравнения моделей на одинаковом масштабе входа и ради скорости обучения
IMAGE_SIZE = {"shortest_edge": 480, "longest_edge": 640}


def get_detr(num_classes: int):
    """
    Возвращает (model, processor).

    processor нужен и для обучения (через collate_fn), и для инференса —
    храните его вместе с чекпоинтом модели.
    """
    processor = DetrImageProcessor.from_pretrained(CHECKPOINT, size=IMAGE_SIZE)

    model = DetrForObjectDetection.from_pretrained(
        CHECKPOINT,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    return model, processor