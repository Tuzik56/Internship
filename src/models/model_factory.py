"""
Фабрика моделей детекции из torchvision.models.detection.

Все три модели создаются единообразно: get_model(name, num_classes),
где num_classes НЕ включает фон (фон добавляется автоматически внутри).
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def get_faster_rcnn(num_classes: int, min_size: int = 480, max_size: int = 640):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=min_size, max_size=max_size
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # +1: torchvision резервирует индекс 0 под фон
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


def get_ssd(num_classes: int):
    model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")

    in_channels = [module.in_channels for module in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
    )

    return model


def get_retinanet(num_classes: int, min_size: int = 480, max_size: int = 640):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights="DEFAULT", min_size=min_size, max_size=max_size
    )

    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
    )

    return model


MODEL_REGISTRY = {
    "faster_rcnn": get_faster_rcnn,
    "ssd": get_ssd,
    "retinanet": get_retinanet,
}


def get_model(name: str, num_classes: int):
    """
    Единая точка создания модели.

    Parameters
    ----------
    name : str
        Одно из: "faster_rcnn", "ssd", "retinanet"
    num_classes : int
        Число классов БЕЗ учёта фона (для KITTI — 6)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестная модель: {name}. Доступны: {list(MODEL_REGISTRY)}")

    return MODEL_REGISTRY[name](num_classes)