from torch.utils.data import DataLoader
from src.dataset.detection_dataset import CocoDetectionDataset, collate_fn
from src.models.model_factory import get_model

dataset = CocoDetectionDataset(
    images_dir="data/raw/kitti/training/image_2",
    annotations_path="data/processed/annotations/train.json",
)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

images, targets = next(iter(loader))
model = get_model("faster_rcnn", num_classes=6)
model.train()

loss_dict = model(list(images), list(targets))
print(loss_dict)  # словарь с loss'ами — если печатается без ошибок, всё стыкуется