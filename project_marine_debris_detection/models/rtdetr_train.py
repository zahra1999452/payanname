# -*- coding: utf-8 -*-
"""
RT-DETR draft training script (not used for the final reported checkpoint;
see train_rtdetr_l_6gb.py). Kept for reference. Two earlier commented-out
draft revisions were removed from the top of this file during cleanup.

آموزش RT-DETR-L با Resume خودکار - effective batch size = 16
"""

import sys
import torch
from pathlib import Path
from multiprocessing import cpu_count, freeze_support

# ============================================================
# ۱. بهینه‌سازی‌های CUDA (غیرفعال کردن deterministic برای سرعت)
# ============================================================
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

# ============================================================
# ۲. پیدا کردن config.py
# ============================================================
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent

if (_project_root / "config.py").exists():
    sys.path.insert(0, str(_project_root))
else:
    _project_root = _current_file.parent
    if (_project_root / "config.py").exists():
        sys.path.insert(0, str(_project_root))
    else:
        print("❌ config.py یافت نشد!")
        sys.exit(1)

from config import DATASET_PATH, TRAIN_CONFIG
from ultralytics import YOLO


def train_rtdetr():
    """تابع اصلی آموزش با Resume خودکار و effective batch=16"""
    
    # ============================================================
    # ۳. تنظیمات جدید (effective batch = 16)
    # ============================================================
    REAL_BATCH = 4              # تعداد تصاویر روی GPU (همان قبل)
    EFFECTIVE_BATCH = 16        # 🔥 تغییر به ۱۶ (به‌جای ۳۲)
    IMAGE_SIZE = 640
    WORKERS = min(4, cpu_count())   # حداکثر ۴ برای ویندوز
    CACHE_MODE = "disk"             # کش روی دیسک (پایدار)
    DEVICE = 0                      # RTX 3060

    # ============================================================
    # ۴. مسیر چک‌پوینت (همان لاگ قبلی)
    # ============================================================
    SCRIPT_DIR = Path(__file__).parent
    SAVE_DIR = SCRIPT_DIR / "runs" / "detect" / "results" / "RT-DETR" / "rtdetr-r18-6gb"
    LAST_CKPT_PATH = SAVE_DIR / "weights" / "last.pt"

    print("=" * 80)
    print("🚀 شروع آموزش RT-DETR-L (Resume خودکار)")
    print("=" * 80)
    print(f"   📁 چک‌پوینت: {LAST_CKPT_PATH}")
    print(f"   🖥️  Real Batch (GPU): {REAL_BATCH}")
    print(f"   🎯 Effective Batch: {EFFECTIVE_BATCH}  ← تغییر داده شد")
    print(f"   🔄 Gradient Accumulation Steps: {EFFECTIVE_BATCH // REAL_BATCH}")
    print(f"   🧵 Workers: {WORKERS}")
    print(f"   💿 Cache: {CACHE_MODE}")
    print("-" * 80)

    # ============================================================
    # ۵. بارگذاری مدل با Resume
    # ============================================================
    if LAST_CKPT_PATH.exists():
        print(f"✅ چک‌پوینت پیدا شد → ادامه از اپوک قبلی")
        model = YOLO(str(LAST_CKPT_PATH))
        resume_flag = True
    else:
        print("⚠️  چک‌پوینتی نیست → شروع از وزن پیش‌آموزش‌دیده")
        model = YOLO("rtdetr-l.pt")
        resume_flag = False

    # ============================================================
    # ۶. آموزش
    # ============================================================
    data_yaml = DATASET_PATH / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ data.yaml یافت نشد: {data_yaml}")
        return

    print(f"📄 data.yaml: {data_yaml}")
    print("🔄 شروع آموزش...")
    print("=" * 80)

    results = model.train(
        data=str(data_yaml),
        cache=CACHE_MODE,
        epochs=TRAIN_CONFIG.get("epochs", 150),
        batch=REAL_BATCH,
        nbs=EFFECTIVE_BATCH,      # ← اینجا مقدار ۱۶ قرار می‌گیرد
        imgsz=IMAGE_SIZE,
        lr0=TRAIN_CONFIG.get("lr", 0.0001),
        device=DEVICE,
        workers=WORKERS,
        patience=TRAIN_CONFIG.get("patience", 50),
        project="results/RT-DETR",
        name="rtdetr-r18-6gb",
        exist_ok=True,
        save=True,
        save_period=1,
        plots=True,
        verbose=True,
        amp=False,
        resume=resume_flag,
    )

    print("\n✅ آموزش با موفقیت ادامه یافت (یا کامل شد)!")
    print(f"📁 نتایج: {SAVE_DIR}")


# ============================================================
# ۷. نقطه‌ی ورود اصلی (اجباری برای ویندوز)
# ============================================================
if __name__ == "__main__":
    freeze_support()
    train_rtdetr()