# # # -*- coding: utf-8 -*-
# # """
# # ====================================================================================================
# # آموزش RT-DETR-R18 با Ultralytics - بهینه شده با Gradient Accumulation برای GPU 6GB
# # ====================================================================================================
# # ✅ IMAGE_SIZE = 640
# # ✅ batch = 4 (واقعی روی GPU)
# # ✅ nbs = 32 (effective batch size با Gradient Accumulation)
# # ✅ قابلیت Resume از آخرین checkpoint
# # ✅ غیرفعال کردن AMP برای صرفه‌جویی در حافظه
# # ✅ تنظیمات مطابق با پروپوزال (۱۵۰ epoch، AdamW، lr=0.0001)
# # ====================================================================================================
# # """

# # import sys
# # from pathlib import Path

# # # ==================================================================================================
# # # پیدا کردن config.py (قابل اجرا از هر پوشه‌ای)
# # # ==================================================================================================

# # _current_file = Path(__file__).resolve()
# # _project_root = _current_file.parent.parent  # marine_debris_detection

# # # اگر فایل در پوشه models/ است، دو پله بالا می‌رویم
# # if (_project_root / "config.py").exists():
# #     sys.path.insert(0, str(_project_root))
# # else:
# #     # اگر در پوشه اصلی هستیم، یک پله بالا
# #     _project_root = _current_file.parent
# #     if (_project_root / "config.py").exists():
# #         sys.path.insert(0, str(_project_root))
# #     else:
# #         print("❌ فایل config.py در هیچ‌یک از مسیرهای زیر یافت نشد:")
# #         print(f"   - {_current_file.parent.parent / 'config.py'}")
# #         print(f"   - {_current_file.parent / 'config.py'}")
# #         print("\nلطفاً فایل config.py را در پوشه پروژه ایجاد کنید.")
# #         sys.exit(1)

# # # ==================================================================================================
# # # ایمپورت تنظیمات
# # # ==================================================================================================

# # try:
# #     from config import DATASET_PATH, TRAIN_CONFIG
# #     print("✅ config.py با موفقیت بارگذاری شد.")
# # except ImportError as e:
# #     print(f"❌ خطا در بارگذاری config.py: {e}")
# #     sys.exit(1)

# # # ==================================================================================================
# # # ایمپورت Ultralytics
# # # ==================================================================================================

# # from ultralytics import YOLO

# # # ==================================================================================================
# # # تنظیمات بهینه برای RTX 3060 6GB
# # # ==================================================================================================

# # REAL_BATCH = 4           # تعداد تصاویر همزمان روی GPU (مصرف VRAM را تعیین می‌کند)
# # EFFECTIVE_BATCH = 32     # اندازه دسته مؤثر نهایی (Gradient Accumulation)
# # IMAGE_SIZE = 640         # سایز تصویر ورودی (مطابق پروپوزال)
# # WORKERS = 2              # کاهش workers برای صرفه‌جویی در RAM سیستم

# # # ==================================================================================================
# # # تابع اصلی آموزش
# # # ==================================================================================================

# # def train_rtdetr(resume=True, force_restart=False):
# #     """
# #     آموزش RT-DETR با قابلیت Resume
    
# #     Args:
# #         resume (bool): اگر True باشد، از آخرین checkpoint (last.pt) ادامه می‌دهد.
# #                        اگر False باشد، از ابتدا شروع می‌کند.
# #         force_restart (bool): اگر True باشد، حتی اگر checkpoint وجود داشته باشد،
# #                               از ابتدا شروع می‌کند (با حذف پوشه قبلی).
# #     """
# #     print("=" * 80)
# #     print("🚀 آموزش RT-DETR (کوچک‌ترین نسخه پشتیبانی‌شده توسط Ultralytics)")
# #     print("=" * 80)
# #     print(f"   🖥️  Real Batch Size (GPU): {REAL_BATCH}")
# #     print(f"   🎯 Effective Batch Size (nbs): {EFFECTIVE_BATCH}")
# #     print(f"   📐 Image Size: {IMAGE_SIZE}")
# #     print(f"   🔄 Gradient Accumulation Steps: {EFFECTIVE_BATCH // REAL_BATCH}")
# #     print(f"   💾 GPU Memory Target: ~5.5-6 GB")
# #     print("-" * 80)
    
# #     if force_restart:
# #         print("⚠️  حالت Force Restart: شروع از ابتدا (حذف نتایج قبلی)")
# #         resume = False
# #     elif resume:
# #         print("🔄 حالت Resume: ادامه از آخرین checkpoint (در صورت وجود)")
# #     else:
# #         print("🆕 شروع از ابتدا (بدون استفاده از checkpoint قبلی)")
    
# #     print("-" * 80)

# #     # ==============================================================================================
# #     # ۱. بررسی وجود فایل data.yaml
# #     # ==============================================================================================
    
# #     data_yaml = DATASET_PATH / "data.yaml"
# #     if not data_yaml.exists():
# #         print(f"❌ فایل data.yaml در مسیر زیر یافت نشد:")
# #         print(f"   {data_yaml}")
# #         print("\nلطفاً مسیر صحیح را در فایل config.py تنظیم کنید.")
# #         return
    
# #     print(f"📄 فایل data.yaml: {data_yaml}")
    
# #     # ==============================================================================================
# #     # ۲. تعیین مسیر پروژه و چک‌پوینت‌ها
# #     # ==============================================================================================
    
# #     PROJECT_ROOT = Path(__file__).parent.parent  # marine_debris_detection
# #     RESULTS_DIR = PROJECT_ROOT / "results" / "RT-DETR" / "rtdetr-r18-6gb"
# #     WEIGHTS_DIR = RESULTS_DIR / "weights"
# #     LAST_CKPT = WEIGHTS_DIR / "last.pt"
    
# #     print(f"📁 پوشه نتایج: {RESULTS_DIR}")
    
# #     # ==============================================================================================
# #     # ۳. بارگذاری مدل
# #     # ==============================================================================================
    
# #     # اگر force_restart=True باشد، پوشه قبلی را حذف می‌کنیم
# #     if force_restart and RESULTS_DIR.exists():
# #         import shutil
# #         print(f"🗑️  حذف نتایج قبلی از: {RESULTS_DIR}")
# #         shutil.rmtree(RESULTS_DIR)
    
# #     # اگر resume=True و last.pt وجود داشته باشد، از آن بارگذاری می‌کنیم
# #     if resume and LAST_CKPT.exists() and not force_restart:
# #         print(f"📂 ادامه از چک‌پوینت: {LAST_CKPT}")
# #         model = YOLO(str(LAST_CKPT))
# #         resume_flag = True
# #     else:
# #         if resume and not LAST_CKPT.exists() and not force_restart:
# #             print("⚠️  هیچ چک‌پوینت قبلی یافت نشد. از وزن پیش‌آموزش‌دیده شروع می‌شود.")
# #         print("📥 بارگذاری وزن پیش‌آموزش‌دیده: rtdetr-l.pt")
# #         model = YOLO("rtdetr-l.pt")
# #         resume_flag = False
    
# #     # ==============================================================================================
# #     # ۴. اجرای آموزش با Gradient Accumulation
# #     # ==============================================================================================
    
# #     print("\n🔄 شروع آموزش...")
# #     print("-" * 80)
# #     print(f"   🔹 batch = {REAL_BATCH} (واقعی روی GPU)")
# #     print(f"   🔹 nbs = {EFFECTIVE_BATCH} (effective batch size)")
# #     print(f"   🔹 Gradient Accumulation = {EFFECTIVE_BATCH // REAL_BATCH} steps")
# #     print(f"   🔹 Total Epochs = {TRAIN_CONFIG.get('epochs', 150)}")
# #     print("-" * 80)
    
# #     results = model.train(
# #         # ==================== داده‌ها ====================
# #         data=str(data_yaml),
        
# #         # ==================== تنظیمات آموزش ====================
# #         epochs=TRAIN_CONFIG.get("epochs", 150),
        
# #         # 🔻 Gradient Accumulation (کلیدی برای کاهش VRAM)
# #         batch=REAL_BATCH,          # 4 تصویر در هر step (مصرف VRAM پایین)
# #         nbs=EFFECTIVE_BATCH,       # 32 تصویر مؤثر (Gradient Accumulation)
        
# #         imgsz=IMAGE_SIZE,          # 640
# #         lr0=TRAIN_CONFIG.get("lr", 0.0001),
# #         device=0,                  # GPU 0
# #         workers=WORKERS,           # 2 workers برای صرفه‌جویی در RAM
        
# #         patience=TRAIN_CONFIG.get("patience", 50),
        
# #         # ==================== ذخیره‌سازی ====================
# #         project="results/RT-DETR",
# #         name="rtdetr-r18-6gb",
# #         exist_ok=True,
# #         save=True,
# #         save_period=1,   # ✅ تغییر کلیدی: ذخیره چک‌پوینت بعد از هر اپوک
        
# #         # ==================== گزارش و نمودار ====================
# #         plots=True,
# #         verbose=True,
        
# #         # ==================== غیرفعال کردن AMP ====================
# #         amp=False,                 # طبق پروپوزال
        
# #         # ==================== Resume ====================
# #         resume=resume_flag,
# #     )
    
# #     # ==============================================================================================
# #     # ۵. گزارش نهایی
# #     # ==============================================================================================
    
# #     print("\n" + "=" * 80)
# #     print("✅ آموزش RT-DETR با موفقیت کامل شد!")
# #     print("=" * 80)
# #     print(f"📁 نتایج در: {RESULTS_DIR}")
# #     print(f"   - بهترین مدل (best.pt): {WEIGHTS_DIR / 'best.pt'}")
# #     print(f"   - آخرین مدل (last.pt): {WEIGHTS_DIR / 'last.pt'}")
# #     print(f"   - گزارش‌ها و نمودارها: {RESULTS_DIR}")
# #     print("=" * 80)


# # # ==============================================================================================
# # # اجرای مستقیم
# # # ==============================================================================================

# # if __name__ == "__main__":
# #     # ==============================================================================================
# #     # تنظیمات اجرا (قابل تغییر توسط کاربر)
# #     # ==============================================================================================
    
# #     # 1️⃣  ادامه از آخرین checkpoint (پیش‌فرض - توصیه‌شده):
# #     train_rtdetr(resume=True, force_restart=False)
    
# #     # 2️⃣  شروع از ابتدا (بدون استفاده از checkpoint):
# #     # train_rtdetr(resume=False, force_restart=False)
    
# #     # 3️⃣  شروع از ابتدا با حذف نتایج قبلی:
# #     # train_rtdetr(resume=False, force_restart=True)
# # -*- coding: utf-8 -*-
# """
# آموزش RT-DETR-L با Resume خودکار (سازگار با Ultralytics 8.4.37)
# """

# import sys
# import torch
# from pathlib import Path
# from multiprocessing import cpu_count

# # ============================================================
# # ۱. بهینه‌سازی‌های CUDA
# # ============================================================
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
#     torch.use_deterministic_algorithms(False)
#     print("✅ CUDA optimizations applied")

# # ============================================================
# # ۲. پیدا کردن config.py
# # ============================================================
# _current_file = Path(__file__).resolve()
# _project_root = _current_file.parent.parent

# if (_project_root / "config.py").exists():
#     sys.path.insert(0, str(_project_root))
# else:
#     _project_root = _current_file.parent
#     if (_project_root / "config.py").exists():
#         sys.path.insert(0, str(_project_root))
#     else:
#         print("❌ config.py یافت نشد!")
#         sys.exit(1)

# from config import DATASET_PATH, TRAIN_CONFIG
# from ultralytics import YOLO

# # ============================================================
# # ۳. تنظیمات نهایی (بدون پارامترهای ناپشتیبانی)
# # ============================================================
# REAL_BATCH = 4
# EFFECTIVE_BATCH = 32
# IMAGE_SIZE = 640
# WORKERS = min(4, cpu_count())   # حداکثر ۴ برای ویندوز
# CACHE_MODE = "disk"             # کش روی دیسک (سریع و پایدار)
# DEVICE = 0                      # RTX 3060

# # ============================================================
# # ۴. مسیر چک‌پوینت (همان لاگ قبلی)
# # ============================================================
# SCRIPT_DIR = Path(__file__).parent
# SAVE_DIR = SCRIPT_DIR / "runs" / "detect" / "results" / "RT-DETR" / "rtdetr-r18-6gb"
# LAST_CKPT_PATH = SAVE_DIR / "weights" / "last.pt"

# print("=" * 80)
# print("🚀 شروع آموزش RT-DETR-L (Resume خودکار)")
# print("=" * 80)
# print(f"   📁 چک‌پوینت: {LAST_CKPT_PATH}")
# print(f"   🖥️  Batch: {REAL_BATCH} (effective {EFFECTIVE_BATCH})")
# print(f"   🧵 Workers: {WORKERS}")
# print(f"   💿 Cache: {CACHE_MODE}")
# print("-" * 80)

# # ============================================================
# # ۵. بارگذاری مدل با Resume
# # ============================================================
# if LAST_CKPT_PATH.exists():
#     print(f"✅ چک‌پوینت پیدا شد → ادامه از اپوک قبلی")
#     model = YOLO(str(LAST_CKPT_PATH))
#     resume_flag = True
# else:
#     print("⚠️  چک‌پوینتی نیست → شروع از وزن پیش‌آموزش‌دیده")
#     model = YOLO("rtdetr-l.pt")
#     resume_flag = False

# # ============================================================
# # ۶. آموزش
# # ============================================================
# data_yaml = DATASET_PATH / "data.yaml"
# if not data_yaml.exists():
#     print(f"❌ data.yaml یافت نشد: {data_yaml}")
#     sys.exit(1)

# print(f"📄 data.yaml: {data_yaml}")
# print("🔄 شروع آموزش...")
# print("=" * 80)

# results = model.train(
#     data=str(data_yaml),
#     cache=CACHE_MODE,
#     epochs=TRAIN_CONFIG.get("epochs", 150),
#     batch=REAL_BATCH,
#     nbs=EFFECTIVE_BATCH,
#     imgsz=IMAGE_SIZE,
#     lr0=TRAIN_CONFIG.get("lr", 0.0001),
#     device=DEVICE,
#     workers=WORKERS,
#     patience=TRAIN_CONFIG.get("patience", 50),
#     project="results/RT-DETR",
#     name="rtdetr-r18-6gb",
#     exist_ok=True,
#     save=True,
#     save_period=1,
#     plots=True,
#     verbose=True,
#     amp=False,
#     resume=resume_flag,
# )

# print("\n✅ آموزش با موفقیت ادامه یافت (یا کامل شد)!")
# print(f"📁 نتایج: {SAVE_DIR}")
# -*- coding: utf-8 -*-
"""
آموزش RT-DETR-L با Resume خودکار - سازگار با ویندوز
"""

# import sys
# import torch
# from pathlib import Path
# from multiprocessing import cpu_count, freeze_support

# # ============================================================
# # ۱. بهینه‌سازی‌های CUDA
# # ============================================================
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
#     torch.use_deterministic_algorithms(False)

# # ============================================================
# # ۲. پیدا کردن config.py
# # ============================================================
# _current_file = Path(__file__).resolve()
# _project_root = _current_file.parent.parent

# if (_project_root / "config.py").exists():
#     sys.path.insert(0, str(_project_root))
# else:
#     _project_root = _current_file.parent
#     if (_project_root / "config.py").exists():
#         sys.path.insert(0, str(_project_root))
#     else:
#         print("❌ config.py یافت نشد!")
#         sys.exit(1)

# from config import DATASET_PATH, TRAIN_CONFIG
# from ultralytics import YOLO


# def train_rtdetr():
#     """تابع اصلی آموزش با Resume خودکار"""
    
#     # ============================================================
#     # ۳. تنظیمات نهایی (بدون پارامترهای ناپشتیبانی)
#     # ============================================================
#     REAL_BATCH = 4
#     EFFECTIVE_BATCH = 32
#     IMAGE_SIZE = 640
#     WORKERS = min(4, cpu_count())   # حداکثر ۴ برای ویندوز
#     CACHE_MODE = "disk"             # کش روی دیسک (سریع و پایدار)
#     DEVICE = 0                      # RTX 3060

#     # ============================================================
#     # ۴. مسیر چک‌پوینت (همان لاگ قبلی)
#     # ============================================================
#     SCRIPT_DIR = Path(__file__).parent
#     SAVE_DIR = SCRIPT_DIR / "runs" / "detect" / "results" / "RT-DETR" / "rtdetr-r18-6gb"
#     LAST_CKPT_PATH = SAVE_DIR / "weights" / "last.pt"

#     print("=" * 80)
#     print("🚀 شروع آموزش RT-DETR-L (Resume خودکار)")
#     print("=" * 80)
#     print(f"   📁 چک‌پوینت: {LAST_CKPT_PATH}")
#     print(f"   🖥️  Batch: {REAL_BATCH} (effective {EFFECTIVE_BATCH})")
#     print(f"   🧵 Workers: {WORKERS}")
#     print(f"   💿 Cache: {CACHE_MODE}")
#     print("-" * 80)

#     # ============================================================
#     # ۵. بارگذاری مدل با Resume
#     # ============================================================
#     if LAST_CKPT_PATH.exists():
#         print(f"✅ چک‌پوینت پیدا شد → ادامه از اپوک قبلی")
#         model = YOLO(str(LAST_CKPT_PATH))
#         resume_flag = True
#     else:
#         print("⚠️  چک‌پوینتی نیست → شروع از وزن پیش‌آموزش‌دیده")
#         model = YOLO("rtdetr-l.pt")
#         resume_flag = False

#     # ============================================================
#     # ۶. آموزش
#     # ============================================================
#     data_yaml = DATASET_PATH / "data.yaml"
#     if not data_yaml.exists():
#         print(f"❌ data.yaml یافت نشد: {data_yaml}")
#         return

#     print(f"📄 data.yaml: {data_yaml}")
#     print("🔄 شروع آموزش...")
#     print("=" * 80)

#     results = model.train(
#         data=str(data_yaml),
#         cache=CACHE_MODE,
#         epochs=TRAIN_CONFIG.get("epochs", 150),
#         batch=REAL_BATCH,
#         nbs=EFFECTIVE_BATCH,
#         imgsz=IMAGE_SIZE,
#         lr0=TRAIN_CONFIG.get("lr", 0.0001),
#         device=DEVICE,
#         workers=WORKERS,
#         patience=TRAIN_CONFIG.get("patience", 50),
#         project="results/RT-DETR",
#         name="rtdetr-r18-6gb",
#         exist_ok=True,
#         save=True,
#         save_period=1,
#         plots=True,
#         verbose=True,
#         amp=False,
#         resume=resume_flag,
#     )

#     print("\n✅ آموزش با موفقیت ادامه یافت (یا کامل شد)!")
#     print(f"📁 نتایج: {SAVE_DIR}")


# # ============================================================
# # ۷. نقطه‌ی ورود اصلی (اجباری برای ویندوز)
# # ============================================================
# if __name__ == "__main__":
#     freeze_support()   # برای جلوگیری از خطای spawn در ویندوز
#     train_rtdetr()
# -*- coding: utf-8 -*-
"""
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