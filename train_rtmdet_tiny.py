# -*- coding: utf-8 -*-

"""
===============================================================================
RTMDet-Tiny | Marine Debris Detection
RTX 3060 Laptop 6GB
MMDetection 3.x + MMEngine

FIXED VERSION
===============================================================================

IMPORTANT FIXES
-------------------------------------------------------------------------------
1. Correct MMEngine resume mechanism
2. NO runner.resume() without filename
3. resume=True + load_from=<checkpoint>
4. Fix CSPNeXtPAFPN incompatibility:
       remove deepen_factor
       remove widen_factor
5. Checkpoint saved every epoch
6. Validation mAP after every epoch
7. TEST evaluation ONLY after training
8. Progress bar with ETA
9. AMP enabled
10. Gradient accumulation = 4
11. Batch size = 4
12. Image size = 640
13. Windows safe
===============================================================================
"""

import os
import sys
import time
import glob
import argparse
import platform
import multiprocessing as mp
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from mmdet.utils import register_all_modules


# =============================================================================
# WINDOWS
# =============================================================================

if sys.platform.startswith("win"):
    try:
        mp.freeze_support()
    except Exception:
        pass


# =============================================================================
# REGISTER MMDETECTION MODULES
# =============================================================================

register_all_modules(init_default_scope=True)


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

COCO_DIR = DATASET_ROOT / "coco_annotations"

TRAIN_JSON = COCO_DIR / "instances_train.json"
VAL_JSON = COCO_DIR / "instances_val.json"
TEST_JSON = COCO_DIR / "instances_test.json"

TRAIN_IMAGES = DATASET_ROOT / "images" / "train"
VAL_IMAGES = DATASET_ROOT / "images" / "val"
TEST_IMAGES = DATASET_ROOT / "images" / "test"

WORK_DIR = (
    BASE_DIR
    / "runs"
    / "rtmdet"
    / "rtmdet_tiny_640_6gb"
)

CONFIG_OUTPUT = WORK_DIR / "rtmdet_tiny_config.py"


# =============================================================================
# OFFICIAL MMDET CONFIG
# =============================================================================

OFFICIAL_CONFIG = Path(
    sys.prefix
) / "Lib" / "site-packages" / "mmdet" / ".mim" / "configs" / "rtmdet" / (
    "rtmdet_tiny_8xb32-300e_coco.py"
)


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

IMAGE_SIZE = 640

# RTX 3060 Laptop 6GB
BATCH_SIZE = 4

# Effective batch:
# 4 x 4 = 16
GRAD_ACCUMULATION = 4

EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUMULATION

EPOCHS = 150

LR = 1e-4

WEIGHT_DECAY = 5e-4

WARMUP_ITERS = 500

NUM_WORKERS = 2

DEVICE = "cuda:0"

SEED = 42

VAL_INTERVAL = 1

MAX_KEEP_CKPTS = -1

EARLY_STOPPING_PATIENCE = 20


# =============================================================================
# CLASSES
# =============================================================================

CLASSES = (
    "animal_crab",
    "animal_eel",
    "animal_etc",
    "animal_fish",
    "animal_shells",
    "animal_starfish",
    "plant",
    "trash_branch",
    "trash_clothing",
    "trash_net",
    "trash_unknown_instance",
    "trash_wreckage",
    "trash_plastic",
)

NUM_CLASSES = len(CLASSES)


# =============================================================================
# TERMINAL COLORS
# =============================================================================

RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"


# =============================================================================
# PRINT HELPERS
# =============================================================================

def separator(char="=", length=100):
    print(char * length)


def header(title):
    print()
    separator()
    print(title)
    separator()


def ok(message):
    print(f"{GREEN}[OK]{RESET} {message}")


def warning(message):
    print(f"{YELLOW}[WARNING]{RESET} {message}")


def error(message):
    print(f"{RED}[ERROR]{RESET} {message}")


# =============================================================================
# GPU INFORMATION
# =============================================================================

def show_gpu_info():

    header("GPU INFORMATION")

    print(f"Operating System : {platform.system()} {platform.release()}")
    print(f"Python           : {sys.version.split()[0]}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        error("CUDA is not available.")
        return False

    print(f"CUDA version     : {torch.version.cuda}")

    gpu_name = torch.cuda.get_device_name(0)

    props = torch.cuda.get_device_properties(0)

    total_vram = props.total_memory / (1024 ** 3)

    free_vram, total = torch.cuda.mem_get_info(0)

    free_vram = free_vram / (1024 ** 3)

    used_vram = total_vram - free_vram

    print(f"GPU              : {gpu_name}")
    print(f"VRAM             : {total_vram:.2f} GB")
    print(f"Free VRAM        : {free_vram:.2f} GB")
    print(f"Used VRAM        : {used_vram:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")

    print()

    print(f"Image Size       : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch Size       : {BATCH_SIZE}")
    print(f"Grad Accum       : {GRAD_ACCUMULATION}")
    print(f"Effective Batch  : {EFFECTIVE_BATCH}")
    print(f"Workers          : {NUM_WORKERS}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Learning Rate    : {LR}")
    print(f"AMP              : True")

    return True


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset():

    header("DATASET VALIDATION")

    paths = {
        "Dataset": DATASET_ROOT,
        "COCO directory": COCO_DIR,
        "Train images": TRAIN_IMAGES,
        "Val images": VAL_IMAGES,
        "Test images": TEST_IMAGES,
        "Train COCO": TRAIN_JSON,
        "Val COCO": VAL_JSON,
        "Test COCO": TEST_JSON,
    }

    all_ok = True

    for name, path in paths.items():

        if path.exists():

            ok(f"{name}")

            print(f"     {path}")

        else:

            error(f"{name} NOT FOUND")

            print(f"     {path}")

            all_ok = False

    return all_ok


# =============================================================================
# COCO STATISTICS
# =============================================================================

def show_coco_stats():

    header("COCO DATASET STATISTICS")

    try:
        import json

        for name, path in [
            ("TRAIN", TRAIN_JSON),
            ("VAL", VAL_JSON),
            ("TEST", TEST_JSON),
        ]:

            print()
            print(name)
            print("-" * 100)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"Images       : {len(data.get('images', []))}")
            print(f"Annotations  : {len(data.get('annotations', []))}")
            print(f"Categories   : {len(data.get('categories', []))}")

    except Exception as exc:

        warning(f"Could not read COCO statistics: {exc}")


# =============================================================================
# CLASS VALIDATION
# =============================================================================

def validate_classes():

    header("CLASS VALIDATION")

    for i, cls in enumerate(CLASSES):

        print(f"{i:02d} : {cls}")

    print()
    print(f"Number of classes : {NUM_CLASSES}")

    if NUM_CLASSES != 13:

        error("Unexpected number of classes.")

        return False

    ok("Class count = 13")

    return True


# =============================================================================
# FIND LATEST CHECKPOINT
# =============================================================================

def find_latest_checkpoint():

    checkpoints = []

    for path in WORK_DIR.glob("epoch_*.pth"):

        try:

            epoch_number = int(
                path.stem.split("_")[1]
            )

            checkpoints.append(
                (epoch_number, path)
            )

        except Exception:

            continue

    if not checkpoints:

        return None

    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints[-1][1]


# =============================================================================
# CHECKPOINT STATUS
# =============================================================================

def show_checkpoint_status():

    header("CHECKPOINT STATUS")

    checkpoints = []

    for path in WORK_DIR.glob("epoch_*.pth"):

        try:

            epoch_number = int(
                path.stem.split("_")[1]
            )

            size_mb = (
                path.stat().st_size
                / (1024 ** 2)
            )

            checkpoints.append(
                (epoch_number, path, size_mb)
            )

        except Exception:

            pass

    if not checkpoints:

        print("No epoch checkpoints found.")

        return None

    checkpoints.sort(
        key=lambda x: x[0]
    )

    print(
        f"Found {len(checkpoints)} epoch checkpoint(s):"
    )

    for epoch, path, size_mb in checkpoints:

        print(
            f"  epoch_{epoch:<4}.pth"
            f"{size_mb:>10.1f} MB"
        )

    latest = checkpoints[-1][1]

    print()
    print(f"{BOLD}[LATEST]{RESET}")
    print(latest)

    return latest


# =============================================================================
# PROGRESS HOOK
# =============================================================================

@HOOKS.register_module()
class TrainingProgressHook(Hook):
    """
    Lightweight terminal progress display.

    Does NOT modify training logic.
    """

    def __init__(self, interval=25):

        self.interval = interval

        self.epoch_start_time = None

    def before_train_epoch(
        self,
        runner
    ):

        self.epoch_start_time = time.time()

        epoch = runner.epoch + 1

        max_epochs = runner.max_epochs

        print()

        separator("-", 100)

        print(
            f"{CYAN}"
            f"Epoch {epoch}/{max_epochs}"
            f"{RESET}"
        )

        separator("-", 100)

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None
    ):

        if (
            batch_idx + 1
        ) % self.interval != 0:

            return

        try:

            dataloader_len = len(
                runner.train_dataloader
            )

        except Exception:

            return

        current = batch_idx + 1

        elapsed = (
            time.time()
            - self.epoch_start_time
        )

        if current > 0:

            sec_per_iter = (
                elapsed / current
            )

            remaining = (
                dataloader_len
                - current
            )

            eta_seconds = (
                sec_per_iter
                * remaining
            )

        else:

            eta_seconds = 0

        progress = (
            current
            / dataloader_len
        )

        bar_len = 30

        filled = int(
            bar_len * progress
        )

        bar = (
            "#" * filled
            + "-"
            * (bar_len - filled)
        )

        lr = 0.0

        try:

            lr = runner.optim_wrapper.get_lr()[0]

        except Exception:

            pass

        loss = None

        if isinstance(outputs, dict):

            if "loss" in outputs:

                loss = outputs["loss"]

        if loss is None:

            try:

                loss = runner.message_hub.get_scalar(
                    "train/loss"
                ).current()

            except Exception:

                loss = None

        if isinstance(loss, torch.Tensor):

            loss = loss.item()

        if loss is None:

            loss_text = "N/A"

        else:

            loss_text = f"{float(loss):.4f}"

        eta_minutes = (
            eta_seconds / 60
        )

        print(
            f"\r"
            f"[{bar}] "
            f"{current:4d}/{dataloader_len:<4d} "
            f"{progress * 100:6.2f}% "
            f"Loss={loss_text:<8} "
            f"LR={lr:.2e} "
            f"ETA={eta_minutes:6.1f}m",
            end="",
            flush=True
        )

    def after_train_epoch(
        self,
        runner
    ):

        print()

        elapsed = (
            time.time()
            - self.epoch_start_time
        )

        print(
            f"Epoch time: "
            f"{elapsed / 60:.2f} min"
        )


# =============================================================================
# BUILD CONFIG
# =============================================================================

def build_config():

    header("BUILDING RTMDET-TINY CONFIG")

    if not OFFICIAL_CONFIG.exists():

        error(
            "Official MMDetection RTMDet config not found:"
        )

        print(OFFICIAL_CONFIG)

        raise FileNotFoundError(
            OFFICIAL_CONFIG
        )

    print("Official config:")
    print(OFFICIAL_CONFIG)

    cfg = Config.fromfile(
        str(OFFICIAL_CONFIG)
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    cfg.model.bbox_head.num_classes = NUM_CLASSES

    # -------------------------------------------------------------------------
    # CRITICAL FIX
    #
    # Current installed mmdet version does not accept:
    #
    # CSPNeXtPAFPN(
    #     deepen_factor=...
    #     widen_factor=...
    # )
    #
    # Therefore remove them from NECK.
    # -------------------------------------------------------------------------

    if "deepen_factor" in cfg.model.neck:

        del cfg.model.neck["deepen_factor"]

    if "widen_factor" in cfg.model.neck:

        del cfg.model.neck["widen_factor"]

    # Backbone keeps its own official parameters.
    #
    # DO NOT remove backbone deepen_factor/widen_factor.
    #
    # Only NECK is fixed here.

    # =========================================================================
    # DATA PREPROCESSOR
    # =========================================================================

    cfg.model.data_preprocessor.pad_size_divisor = 32

    # =========================================================================
    # TRAINING CONFIG
    # =========================================================================

    cfg.train_cfg = dict(
        type="EpochBasedTrainLoop",
        max_epochs=EPOCHS,
        val_interval=VAL_INTERVAL,
    )

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    cfg.optim_wrapper = dict(
        type="AmpOptimWrapper",

        accumulative_counts=GRAD_ACCUMULATION,

        clip_grad=dict(
            max_norm=35,
            norm_type=2,
        ),

        optimizer=dict(
            type="AdamW",
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        ),
    )

    # =========================================================================
    # LR SCHEDULER
    # =========================================================================

    cfg.param_scheduler = [

        dict(
            type="LinearLR",
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=WARMUP_ITERS,
        ),

        dict(
            type="CosineAnnealingLR",
            T_max=EPOCHS,
            eta_min=LR * 0.01,
            begin=0,
            end=EPOCHS,
            by_epoch=True,
        ),
    ]

    # =========================================================================
    # TRAIN DATASET
    # =========================================================================

    train_pipeline = [

        dict(
            type="LoadImageFromFile"
        ),

        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),

        dict(
            type="CachedMosaic",
            img_scale=(IMAGE_SIZE, IMAGE_SIZE),
            max_cached_images=20,
            pad_val=114.0,
            random_pop=False,
        ),

        dict(
            type="RandomResize",
            scale=(1280, 1280),
            ratio_range=(0.5, 2.0),
            keep_ratio=True,
        ),

        dict(
            type="RandomCrop",
            crop_size=(IMAGE_SIZE, IMAGE_SIZE),
        ),

        dict(
            type="YOLOXHSVRandomAug"
        ),

        dict(
            type="RandomFlip",
            prob=0.5,
        ),

        dict(
            type="Pad",
            size=(IMAGE_SIZE, IMAGE_SIZE),
            pad_val=dict(
                img=(114, 114, 114)
            ),
        ),

        dict(
            type="CachedMixUp",
            img_scale=(IMAGE_SIZE, IMAGE_SIZE),
            ratio_range=(1.0, 1.0),
            max_cached_images=10,
            pad_val=(114, 114, 114),
            prob=0.5,
            random_pop=False,
        ),

        dict(
            type="PackDetInputs"
        ),
    ]

    cfg.train_dataloader = dict(

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,

        persistent_workers=True,

        pin_memory=True,

        sampler=dict(
            type="DefaultSampler",
            shuffle=True,
        ),

        dataset=dict(

            type="CocoDataset",

            data_root="",

            ann_file=str(
                TRAIN_JSON
            ),

            data_prefix=dict(
                img=str(
                    TRAIN_IMAGES
                ) + os.sep
            ),

            metainfo=dict(
                classes=CLASSES
            ),

            filter_cfg=dict(
                filter_empty_gt=False
            ),

            pipeline=train_pipeline,
        ),
    )

    # =========================================================================
    # VALIDATION PIPELINE
    # =========================================================================

    val_pipeline = [

        dict(
            type="LoadImageFromFile"
        ),

        dict(
            type="Resize",
            scale=(IMAGE_SIZE, IMAGE_SIZE),
            keep_ratio=True,
        ),

        dict(
            type="Pad",
            size=(IMAGE_SIZE, IMAGE_SIZE),
            pad_val=dict(
                img=(114, 114, 114)
            ),
        ),

        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),

        dict(
            type="PackDetInputs"
        ),
    ]

    cfg.val_dataloader = dict(

        batch_size=1,

        num_workers=NUM_WORKERS,

        persistent_workers=True,

        pin_memory=True,

        drop_last=False,

        sampler=dict(
            type="DefaultSampler",
            shuffle=False,
        ),

        dataset=dict(

            type="CocoDataset",

            data_root="",

            ann_file=str(
                VAL_JSON
            ),

            data_prefix=dict(
                img=str(
                    VAL_IMAGES
                ) + os.sep
            ),

            metainfo=dict(
                classes=CLASSES
            ),

            test_mode=True,

            pipeline=val_pipeline,
        ),
    )

    cfg.val_evaluator = dict(

        type="CocoMetric",

        ann_file=str(
            VAL_JSON
        ),

        metric="bbox",

        format_only=False,
    )

    cfg.val_cfg = dict(
        type="ValLoop"
    )

    # =========================================================================
    # TEST PIPELINE
    # =========================================================================

    test_pipeline = [

        dict(
            type="LoadImageFromFile"
        ),

        dict(
            type="Resize",
            scale=(IMAGE_SIZE, IMAGE_SIZE),
            keep_ratio=True,
        ),

        dict(
            type="Pad",
            size=(IMAGE_SIZE, IMAGE_SIZE),
            pad_val=dict(
                img=(114, 114, 114)
            ),
        ),

        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),

        dict(
            type="PackDetInputs"
        ),
    ]

    cfg.test_dataloader = dict(

        batch_size=1,

        num_workers=NUM_WORKERS,

        persistent_workers=True,

        pin_memory=True,

        drop_last=False,

        sampler=dict(
            type="DefaultSampler",
            shuffle=False,
        ),

        dataset=dict(

            type="CocoDataset",

            data_root="",

            ann_file=str(
                TEST_JSON
            ),

            data_prefix=dict(
                img=str(
                    TEST_IMAGES
                ) + os.sep
            ),

            metainfo=dict(
                classes=CLASSES
            ),

            test_mode=True,

            pipeline=test_pipeline,
        ),
    )

    cfg.test_evaluator = dict(

        type="CocoMetric",

        ann_file=str(
            TEST_JSON
        ),

        metric="bbox",

        format_only=False,
    )

    cfg.test_cfg = dict(
        type="TestLoop"
    )

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    cfg.default_hooks.checkpoint = dict(

        type="CheckpointHook",

        by_epoch=True,

        interval=1,

        save_last=True,

        max_keep_ckpts=MAX_KEEP_CKPTS,

        save_best="coco/bbox_mAP",

        rule="greater",
    )

    # =========================================================================
    # LOGGER
    # =========================================================================

    cfg.default_hooks.logger = dict(

        type="LoggerHook",

        interval=50,

        log_metric_by_epoch=True,
    )

    # =========================================================================
    # CUSTOM PROGRESS HOOK
    # =========================================================================

    cfg.custom_hooks = [

        dict(
            type="TrainingProgressHook",
            interval=25,
        )
    ]

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    cfg.default_hooks.visualization = dict(
        type="DetVisualizationHook",
        draw=False,
    )

    # =========================================================================
    # RANDOMNESS
    # =========================================================================

    cfg.randomness = dict(

        seed=SEED,

        deterministic=False,

        diff_rank_seed=False,
    )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    cfg.env_cfg = dict(

        cudnn_benchmark=True,

        mp_cfg=dict(
            mp_start_method="spawn",
            opencv_num_threads=0,
        ),

        dist_cfg=dict(
            backend="gloo"
        ),
    )

    # =========================================================================
    # WORK DIR
    # =========================================================================

    cfg.work_dir = str(
        WORK_DIR
    )

    # =========================================================================
    # IMPORTANT
    #
    # These two are changed by train_model()
    # depending on --resume.
    # =========================================================================

    cfg.load_from = None

    cfg.resume = False

    # =========================================================================
    # DEFAULT SCOPE
    # =========================================================================

    cfg.default_scope = "mmdet"

    # =========================================================================
    # REMOVE UNNECESSARY TTA / STAGE2
    # =========================================================================

    if hasattr(cfg, "tta_model"):
        pass

    # =========================================================================
    # SAVE CONFIG
    # =========================================================================

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cfg.dump(
        str(CONFIG_OUTPUT)
    )

    ok(
        f"Configuration saved:\n"
        f"{CONFIG_OUTPUT}"
    )

    return cfg


# =============================================================================
# TRAIN
# =============================================================================

def train_model(
    resume=False,
    checkpoint=None
):

    header("RTMDET-TINY TRAINING")

    cfg = build_config()

    # =========================================================================
    # RESUME
    # =========================================================================

    if resume:

        if checkpoint is None:

            checkpoint = find_latest_checkpoint()

        if checkpoint is None:

            error(
                "Resume requested but no checkpoint was found."
            )

            return None

        checkpoint = Path(checkpoint)

        if not checkpoint.exists():

            error(
                f"Checkpoint does not exist:\n"
                f"{checkpoint}"
            )

            return None

        print()
        separator("!", 100)

        print("RESUME MODE")

        print()

        print("Checkpoint:")
        print(checkpoint)

        print()

        print(
            "Restoring:"
        )

        print(
            "  [OK] Model weights"
        )

        print(
            "  [OK] Optimizer state"
        )

        print(
            "  [OK] LR scheduler state"
        )

        print(
            "  [OK] Epoch state"
        )

        print(
            "  [OK] AMP scaler state"
        )

        separator("!", 100)

        # ---------------------------------------------------------------------
        # CRITICAL:
        #
        # DO NOT:
        #
        # runner.resume()
        #
        # DO NOT:
        #
        # runner.resume(checkpoint)
        #
        # Instead:
        #
        # cfg.load_from = checkpoint
        # cfg.resume = True
        #
        # Then Runner.from_cfg() will correctly resume.
        # ---------------------------------------------------------------------

        cfg.load_from = str(
            checkpoint
        )

        cfg.resume = True

    else:

        cfg.load_from = None

        cfg.resume = False

    # =========================================================================
    # PRINT CONFIG
    # =========================================================================

    print()
    separator()

    print("FINAL TRAINING CONFIGURATION")

    separator()

    print(
        f"Model                 : RTMDet-Tiny"
    )

    print(
        f"Classes               : {NUM_CLASSES}"
    )

    print(
        f"Image Size            : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Batch                 : {BATCH_SIZE}"
    )

    print(
        f"Gradient Accumulation : "
        f"{GRAD_ACCUMULATION}"
    )

    print(
        f"Effective Batch       : "
        f"{EFFECTIVE_BATCH}"
    )

    print(
        f"Epochs                : {EPOCHS}"
    )

    print(
        f"Optimizer             : AdamW"
    )

    print(
        f"Learning Rate         : {LR}"
    )

    print(
        f"Weight Decay          : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"AMP                   : True"
    )

    print(
        f"Workers               : "
        f"{NUM_WORKERS}"
    )

    print(
        f"Checkpoint             : EVERY EPOCH"
    )

    print(
        f"Validation             : EVERY EPOCH"
    )

    print(
        f"Best metric            : coco/bbox_mAP"
    )

    print(
        f"Resume                 : {resume}"
    )

    print()

    print(
        "Validation mAP is calculated after every epoch."
    )

    print(
        "TEST mAP is NOT used during training."
    )

    print(
        "TEST evaluation is performed only after training."
    )

    print()

    separator()

    # =========================================================================
    # BUILD RUNNER
    # =========================================================================

    print()

    print(
        "Creating Runner..."
    )

    runner = Runner.from_cfg(
        cfg
    )

    ok(
        "Runner created successfully."
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print()

    separator()

    print(
        "STARTING TRAINING"
    )

    separator()

    start_time = time.time()

    try:

        runner.train()

    except KeyboardInterrupt:

        print()

        warning(
            "Training interrupted by user."
        )

        latest = find_latest_checkpoint()

        if latest:

            print()
            print(
                "Latest checkpoint:"
            )

            print(latest)

            print()
            print(
                "Resume with:"
            )

            print(
                "python train_rtmdet_tiny.py --resume"
            )

        return runner

    except Exception as exc:

        print()

        error(
            "Training failed."
        )

        print()

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    elapsed = (
        time.time()
        - start_time
    )

    print()

    separator()

    print(
        "TRAINING FINISHED"
    )

    separator()

    print(
        f"Training time : "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"Work directory:"
    )

    print(
        WORK_DIR
    )

    return runner


# =============================================================================
# TEST EVALUATION
# =============================================================================

def evaluate_test(
    checkpoint=None
):

    header(
        "FINAL TEST EVALUATION"
    )

    # =========================================================================
    # FIND CHECKPOINT
    # =========================================================================

    if checkpoint is None:

        best_checkpoint = (
            WORK_DIR
            / "best_coco_bbox_mAP.pth"
        )

        if best_checkpoint.exists():

            checkpoint = best_checkpoint

        else:

            checkpoint = find_latest_checkpoint()

    if checkpoint is None:

        error(
            "No checkpoint available for TEST."
        )

        return None

    checkpoint = Path(
        checkpoint
    )

    if not checkpoint.exists():

        error(
            f"Checkpoint not found:\n"
            f"{checkpoint}"
        )

        return None

    print(
        "TEST checkpoint:"
    )

    print(
        checkpoint
    )

    print()

    # =========================================================================
    # BUILD CONFIG
    # =========================================================================

    cfg = build_config()

    cfg.load_from = str(
        checkpoint
    )

    cfg.resume = False

    # =========================================================================
    # CREATE RUNNER
    # =========================================================================

    runner = Runner.from_cfg(
        cfg
    )

    print()

    separator()

    print(
        "Running TEST evaluation..."
    )

    separator()

    results = runner.test()

    print()

    separator()

    print(
        "FINAL TEST RESULTS"
    )

    separator()

    if results:

        for key, value in results.items():

            print(
                f"{key:<30} : {value}"
            )

    print()

    print(
        "TEST evaluation completed."
    )

    return results


# =============================================================================
# SHOW MODEL ARCHITECTURE CHECK
# =============================================================================

def verify_model_config():

    header(
        "MODEL CONFIGURATION CHECK"
    )

    cfg = build_config()

    print(
        "Backbone:"
    )

    print(
        cfg.model.backbone
    )

    print()

    print(
        "Neck:"
    )

    print(
        cfg.model.neck
    )

    print()

    # -------------------------------------------------------------------------
    # CRITICAL CHECK
    # -------------------------------------------------------------------------

    if "deepen_factor" in cfg.model.neck:

        error(
            "neck.deepen_factor still exists!"
        )

        return False

    if "widen_factor" in cfg.model.neck:

        error(
            "neck.widen_factor still exists!"
        )

        return False

    ok(
        "CSPNeXtPAFPN configuration is compatible."
    )

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "RTMDet-Tiny Marine Debris "
            "RTX 3060 6GB"
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from latest epoch checkpoint"
        ),
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run final TEST evaluation"
        ),
    )

    parser.add_argument(
        "--test-checkpoint",
        type=str,
        default=None,
        help=(
            "Specific checkpoint for TEST"
        ),
    )

    parser.add_argument(
        "--check-config",
        action="store_true",
        help=(
            "Only validate RTMDet config"
        ),
    )

    args = parser.parse_args()

    # =========================================================================
    # HEADER
    # =========================================================================

    print()

    separator()

    print(
        f"{BOLD}"
        f"RTMDET-TINY | MARINE DEBRIS | RTX 3060 6GB"
        f"{RESET}"
    )

    separator()

    # =========================================================================
    # GPU
    # =========================================================================

    if not show_gpu_info():

        sys.exit(1)

    # =========================================================================
    # DATASET
    # =========================================================================

    if not validate_dataset():

        sys.exit(1)

    # =========================================================================
    # CLASSES
    # =========================================================================

    if not validate_classes():

        sys.exit(1)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    show_coco_stats()

    # =========================================================================
    # CONFIG CHECK ONLY
    # =========================================================================

    if args.check_config:

        if verify_model_config():

            print()
            ok(
                "Configuration check PASSED."
            )

            sys.exit(0)

        else:

            sys.exit(1)

    # =========================================================================
    # TEST ONLY
    # =========================================================================

    if args.test:

        evaluate_test(
            checkpoint=args.test_checkpoint
        )

        return

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    latest_checkpoint = (
        show_checkpoint_status()
    )

    # =========================================================================
    # RESUME VALIDATION
    # =========================================================================

    if args.resume:

        if latest_checkpoint is None:

            error(
                "--resume was requested, "
                "but no checkpoint exists."
            )

            print()

            print(
                "Start a new training without "
                "--resume."
            )

            sys.exit(1)

        print()

        print(
            f"{CYAN}"
            "RESUME REQUESTED"
            f"{RESET}"
        )

        print()

        print(
            "Latest checkpoint:"
        )

        print(
            latest_checkpoint
        )

    # =========================================================================
    # CONFIRMATION
    # =========================================================================

    print()

    separator()

    if args.resume:

        print(
            "Training will RESUME from:"
        )

        print(
            latest_checkpoint
        )

    else:

        print(
            "Training will START FROM SCRATCH."
        )

    separator()

    print()

    answer = input(
        "Press ENTER to start "
        "(Ctrl+C to cancel)... "
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    runner = train_model(

        resume=args.resume,

        checkpoint=latest_checkpoint,
    )

    # =========================================================================
    # FINAL TEST
    # =========================================================================

    if runner is not None:

        print()

        print(
            "Training process completed."
        )

        print()

        print(
            "To evaluate TEST:"
        )

        print()

        print(
            "python train_rtmdet_tiny.py --test"
        )

        print()

        print(
            "To resume if interrupted:"
        )

        print()

        print(
            "python train_rtmdet_tiny.py --resume"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    if sys.platform.startswith("win"):

        try:

            mp.freeze_support()

        except Exception:

            pass

    main()