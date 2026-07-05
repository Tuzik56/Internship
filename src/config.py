from pathlib import Path

# Корень проекта (поднимаемся на 1 уровень вверх из src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Папки с данными
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# KITTI
KITTI_DIR = RAW_DATA_DIR / "kitti" / "training"
KITTI_IMAGES_DIR = KITTI_DIR / "image_2"
KITTI_LABELS_DIR = KITTI_DIR / "label_2"

# Разбиение датасета
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

# Датасет для YOLO
YOLO_DATASET_DIR = PROCESSED_DATA_DIR / "yolo"

# --- ИСПРАВЛЕННЫЙ БЛОК ДЛЯ МОДЕЛЕЙ ---
# Корневая папка исходного кода
SRC_DIR = PROJECT_ROOT / "src"

# Папка для всех моделей внутри src/
MODELS_DIR = SRC_DIR / "models"

# Специфичные папки для каждой архитектуры
YOLO_MODELS_DIR = MODELS_DIR / "yolo"


VIDEOS_DIR = RAW_DATA_DIR / "videos"

EVALUATION_RUNS_DIR = PROJECT_ROOT / "src" / "evaluation" / "runs"

RESULTS_DIR = PROJECT_ROOT / "results"

SSD_RESULTS_DIR = RESULTS_DIR / "ssd"
SSD_WEIGHTS_DIR = SSD_RESULTS_DIR / "weights"