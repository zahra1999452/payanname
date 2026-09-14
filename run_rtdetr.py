# -*- coding: utf-8 -*-

from pathlib import Path
from ultralytics import YOLO
from config import DATASET_PATH, TRAIN_CONFIG

# مسیرها
DATA_YAML = DATASET_PATH / "data.yaml"

# بارگذاری مدل RT-DETR (نسخه l یا x)
# توجه: Ultralytics از rt-detr-l.pt و rt-detr-x.pt پشتیبانی می‌کند
model = YOLO("rtdetr-l.pt")   # یا "rtdetr-x.pt" برای دقت بالاتر

# آموزش (دقیقاً مثل YOLOv8)
results = model.train(
    data=str(DATA_YAML),
    epochs=TRAIN_CONFIG.get("epochs", 150),
    batch=TRAIN_CONFIG.get("batch_size", 8),
    imgsz=TRAIN_CONFIG.get("img_size", 640),
    lr0=TRAIN_CONFIG.get("lr", 0.0001),
    device=0,
    workers=4,
    patience=50,
    save=True,
    save_period=1,
    plots=True,
    amp=False,
    project="results/RT-DETR",
    name="rtdetr",
    exist_ok=True,
)

print("✅ آموزش RT-DETR کامل شد!")