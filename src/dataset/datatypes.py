from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoundingBox:
    """
    Ограничивающая рамка одного объекта.
    """

    class_name: str

    x1: float
    y1: float

    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def xyxy(self) -> list[float]:
        return [
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        ]

    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class AnnotatedImage:
    """
    Изображение и все объекты, находящиеся на нем.
    """

    image_path: Path

    width: int
    height: int

    boxes: list[BoundingBox]

    @property
    def image_id(self) -> str:
        return self.image_path.stem