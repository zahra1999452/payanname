# -*- coding: utf-8 -*-
# NOTE (repo cleanup): an earlier, fully commented-out draft revision of
# this script (~825 lines) was removed from the top of this file.
# Everything below, including the docstring, is unchanged live code.
"""
================================================================================
RT-DETR-L Training - RTX 3060 Laptop 6GB
Ultralytics
================================================================================

ویژگی‌ها:
- RT-DETR-L
- Image Size = 640
- Batch Size = 6
- AMP = True
- AdamW
- Cosine LR
- Disk Cache
- Windows Safe
- Resume واقعی از last.pt
- حفظ Optimizer / Scheduler / Epoch state
- بدون تغییر Dataset
- بدون تغییر Image Size
================================================================================
"""

from pathlib import Path
import argparse
import sys
import torch

from ultralytics import YOLO


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection"
)

DATASET_ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher\YOLO_DATASETS"
    r"\trashcan_trash_icra19_AUGMENTED_MINOR_ONLY_V6_2_REPAIRED_VALIDATED"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

MODEL_PATH = BASE_DIR / "models" / "rtdetr-l.pt"

OUTPUT_ROOT = BASE_DIR / "runs" / "rtdetr"

RUN_NAME = "rtdetr_l_640_6gb_diskcache"

OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME

WEIGHTS_DIR = OUTPUT_DIR / "weights"

LAST_CHECKPOINT = WEIGHTS_DIR / "last.pt"

BEST_CHECKPOINT = WEIGHTS_DIR / "best.pt"


# =============================================================================
# TRAINING CONFIG
# =============================================================================

IMAGE_SIZE = 640

BATCH_SIZE = 6

# Effective batch size
NBS = 64

EPOCHS = 150

LEARNING_RATE = 1e-4

LR_FINAL_FACTOR = 0.01

WEIGHT_DECAY = 5e-4

MOMENTUM = 0.937

WARMUP_EPOCHS = 2.0

WARMUP_MOMENTUM = 0.8

WARMUP_BIAS_LR = 0.0

WORKERS = 2

DEVICE = 0

AMP = True

COS_LR = True

SEED = 42

PATIENCE = 20

SAVE_PERIOD = 1

CACHE_MODE = "disk"

EXIST_OK = True


# =============================================================================
# AUGMENTATION
# =============================================================================

HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4

DEGREES = 0.0
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 0.0
PERSPECTIVE = 0.0

FLIPUD = 0.0
FLIPLR = 0.5

MOSAIC = 1.0
MIXUP = 0.0
COPY_PASTE = 0.0


# =============================================================================
# SYSTEM CHECK
# =============================================================================

def check_environment():

    print("\n" + "=" * 80)
    print("SYSTEM CHECK")
    print("=" * 80)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(f"CUDA version : {torch.version.cuda}")

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(DEVICE)}"
        )

        props = torch.cuda.get_device_properties(DEVICE)

        vram_gb = props.total_memory / (1024 ** 3)

        print(f"VRAM         : {vram_gb:.2f} GB")

    else:

        raise RuntimeError(
            "CUDA در دسترس نیست. آموزش RT-DETR-L روی CPU اجرا نشود."
        )

    print("=" * 80)


# =============================================================================
# PATH CHECK
# =============================================================================

def check_paths():

    print("\n" + "=" * 80)
    print("PATH CHECK")
    print("=" * 80)

    print(f"Dataset YAML : {DATA_YAML}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Last.pt      : {LAST_CHECKPOINT}")
    print(f"Best.pt      : {BEST_CHECKPOINT}")

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"\nDATA YAML پیدا نشد:\n{DATA_YAML}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nمدل RT-DETR-L پیدا نشد:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n✓ مسیر Dataset صحیح است")
    print("✓ مدل RT-DETR-L پیدا شد")
    print("✓ مسیر خروجی آماده است")

    print("=" * 80)


# =============================================================================
# CHECKPOINT INFORMATION
# =============================================================================

def show_checkpoint_info():

    print("\n" + "=" * 80)
    print("CHECKPOINT CHECK")
    print("=" * 80)

    if not LAST_CHECKPOINT.exists():

        print("⚠ last.pt پیدا نشد.")
        print(
            "در این حالت اگر --resume بزنید، "
            "امکان Resume وجود ندارد."
        )

        return False

    size_mb = LAST_CHECKPOINT.stat().st_size / (1024 * 1024)

    modified = LAST_CHECKPOINT.stat().st_mtime

    print(f"✓ last.pt پیدا شد")
    print(f"  Path : {LAST_CHECKPOINT}")
    print(f"  Size : {size_mb:.2f} MB")

    # تلاش برای بررسی checkpoint
    try:

        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location="cpu",
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            epoch = checkpoint.get("epoch", None)

            if epoch is not None:

                print(
                    f"  Saved Epoch : {epoch}"
                )

                print(
                    f"  Next Epoch  : {epoch + 1}"
                )

            else:

                print(
                    "  ⚠ epoch داخل checkpoint پیدا نشد."
                )

            if "optimizer" in checkpoint:

                print(
                    "  ✓ Optimizer state موجود است"
                )

            else:

                print(
                    "  ⚠ Optimizer state پیدا نشد"
                )

            if "train_args" in checkpoint:

                print(
                    "  ✓ Training arguments موجود است"
                )

        print("=" * 80)

        return True

    except Exception as e:

        print(
            f"⚠ نتوانستم metadata checkpoint را بخوانم: {e}"
        )

        print("=" * 80)

        return True


# =============================================================================
# TRAIN
# =============================================================================

def train_model(
    test_mode=False,
    resume=False
):

    check_environment()

    check_paths()

    # =========================================================================
    # TEST MODE
    # =========================================================================

    if test_mode:

        print("\n" + "=" * 80)
        print("TEST MODE")
        print("=" * 80)

        print("فقط 1 Epoch")
        print("5% Dataset")
        print("برای تست Pipeline")

        model = YOLO(
            str(MODEL_PATH)
        )

        results = model.train(

            data=str(DATA_YAML),

            epochs=1,

            batch=BATCH_SIZE,

            imgsz=IMAGE_SIZE,

            device=DEVICE,

            workers=WORKERS,

            amp=AMP,

            optimizer="AdamW",

            lr0=LEARNING_RATE,

            lrf=LR_FINAL_FACTOR,

            cos_lr=COS_LR,

            weight_decay=WEIGHT_DECAY,

            warmup_epochs=WARMUP_EPOCHS,

            warmup_momentum=WARMUP_MOMENTUM,

            warmup_bias_lr=WARMUP_BIAS_LR,

            momentum=MOMENTUM,

            nbs=NBS,

            patience=1,

            seed=SEED,

            deterministic=True,

            cache=CACHE_MODE,

            save=True,

            save_period=1,

            pretrained=True,

            fraction=0.05,

            project=str(OUTPUT_ROOT),

            name="rtdetr_l_640_test",

            exist_ok=True,

            plots=True,

            verbose=True,

            hsv_h=HSV_H,

            hsv_s=HSV_S,

            hsv_v=HSV_V,

            degrees=DEGREES,

            translate=TRANSLATE,

            scale=SCALE,

            shear=SHEAR,

            perspective=PERSPECTIVE,

            flipud=FLIPUD,

            fliplr=FLIPLR,

            mosaic=MOSAIC,

            mixup=MIXUP,

            copy_paste=COPY_PASTE,

        )

        return results

    # =========================================================================
    # RESUME MODE
    # =========================================================================

    if resume:

        print("\n" + "=" * 80)
        print("RESUME MODE")
        print("=" * 80)

        if not show_checkpoint_info():

            raise FileNotFoundError(
                "\nlast.pt برای Resume پیدا نشد.\n"
                f"Expected:\n{LAST_CHECKPOINT}\n"
            )

        print(
            "\nدر حال بارگذاری checkpoint قبلی:"
        )

        print(
            LAST_CHECKPOINT
        )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # قبلاً اینجا MODEL_PATH یعنی rtdetr-l.pt استفاده می‌شد.
        #
        # rtdetr-l.pt فقط pretrained weights است.
        #
        # برای Resume واقعی باید last.pt لود شود.
        # ---------------------------------------------------------------------

        model = YOLO(
            str(LAST_CHECKPOINT)
        )

        print(
            "\n✓ last.pt به عنوان مدل Resume بارگذاری شد."
        )

        print(
            "✓ Optimizer / Scheduler / Epoch state "
            "از checkpoint بازیابی خواهد شد."
        )

        print(
            f"\nTarget Epochs = {EPOCHS}"
        )

        print(
            "اگر last.pt مربوط به Epoch 25 باشد، "
            "آموزش باید از Epoch 26 ادامه پیدا کند."
        )

        print("=" * 80)

        results = model.train(

            data=str(DATA_YAML),

            # بسیار مهم:
            # EPOCHS هدف نهایی است، نه تعداد epoch جدید
            epochs=EPOCHS,

            batch=BATCH_SIZE,

            imgsz=IMAGE_SIZE,

            device=DEVICE,

            workers=WORKERS,

            amp=AMP,

            cache=CACHE_MODE,

            project=str(OUTPUT_ROOT),

            name=RUN_NAME,

            exist_ok=EXIST_OK,

            save=True,

            save_period=SAVE_PERIOD,

            patience=PATIENCE,

            seed=SEED,

            deterministic=True,

            verbose=True,

            # -----------------------------------------------------------------
            # Resume واقعی
            # -----------------------------------------------------------------

            resume=True,

        )

        return results

    # =========================================================================
    # NEW TRAINING
    # =========================================================================

    print("\n" + "=" * 80)
    print("NEW TRAINING")
    print("=" * 80)

    print(
        "هیچ checkpoint قبلی Resume نمی‌شود."
    )

    print(
        f"Pretrained model:\n{MODEL_PATH}"
    )

    print("=" * 80)

    model = YOLO(
        str(MODEL_PATH)
    )

    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,

        batch=BATCH_SIZE,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        workers=WORKERS,

        amp=AMP,

        optimizer="AdamW",

        lr0=LEARNING_RATE,

        lrf=LR_FINAL_FACTOR,

        cos_lr=COS_LR,

        weight_decay=WEIGHT_DECAY,

        warmup_epochs=WARMUP_EPOCHS,

        warmup_momentum=WARMUP_MOMENTUM,

        warmup_bias_lr=WARMUP_BIAS_LR,

        momentum=MOMENTUM,

        nbs=NBS,

        patience=PATIENCE,

        seed=SEED,

        deterministic=True,

        cache=CACHE_MODE,

        save=True,

        save_period=SAVE_PERIOD,

        pretrained=True,

        project=str(OUTPUT_ROOT),

        name=RUN_NAME,

        exist_ok=EXIST_OK,

        plots=True,

        verbose=True,

        hsv_h=HSV_H,

        hsv_s=HSV_S,

        hsv_v=HSV_V,

        degrees=DEGREES,

        translate=TRANSLATE,

        scale=SCALE,

        shear=SHEAR,

        perspective=PERSPECTIVE,

        flipud=FLIPUD,

        fliplr=FLIPLR,

        mosaic=MOSAIC,

        mixup=MIXUP,

        copy_paste=COPY_PASTE,

    )

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="RT-DETR-L training for RTX 3060 6GB"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run 1-epoch smoke test on 5%% dataset"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from runs/rtdetr/.../weights/last.pt"
    )

    args = parser.parse_args()

    try:

        train_model(

            test_mode=args.test,

            resume=args.resume

        )

    except KeyboardInterrupt:

        print("\n")
        print("=" * 80)
        print("TRAINING INTERRUPTED")
        print("=" * 80)

        print(
            "آموزش توسط کاربر متوقف شد."
        )

        print(
            f"\nCheckpoint احتمالی:\n{LAST_CHECKPOINT}"
        )

        print(
            "\nبرای ادامه دوباره اجرا کنید:"
        )

        print(
            "python .\\train_rtdetr_l_6gb.py --resume"
        )

        print("=" * 80)

        raise

    except Exception as e:

        print("\n")
        print("=" * 80)
        print("TRAINING ERROR")
        print("=" * 80)

        print(
            f"{type(e).__name__}: {e}"
        )

        print("=" * 80)

        raise


if __name__ == "__main__":

    main()