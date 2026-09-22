from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection"
)

PROJECT = ROOT / "hypothesis_tests"

DATASET = Path(
    r"E:\rename\yolo_work_out\resrcher\YOLO_DATASETS"
    r"\trashcan_trash_icra19_AUGMENTED_MINOR_ONLY_V6_2_REPAIRED_VALIDATED"
)

TEST_IMAGES = DATASET / "images" / "test"
TEST_LABELS = DATASET / "labels" / "test"

DATA_YAML = DATASET / "data.yaml"

# ============================================================
# CLASSES
# ============================================================

CLASSES = [
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
]

NUM_CLASSES = 13

# ============================================================
# IMAGE / EVALUATION SETTINGS
# ============================================================

IMAGE_SIZE = 640

IOU_THRESHOLD = 0.50

CONFIDENCE_THRESHOLD = 0.25

# ============================================================
# MODEL CHECKPOINTS
#
# IMPORTANT:
# Replace paths only if the actual final checkpoints differ.
# ============================================================

YOLOV8_WEIGHTS = Path(
    r"E:\rename\yolo_work_out\resrcher\yolo_train"
    r"\results\yolov8n\checkpoints\yolov8\weights\epoch145.pt"
)

RTDETR_WEIGHTS = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection"
    r"\runs\rtdetr\rtdetr_l_640_6gb_diskcache\weights\last.pt"
)

# RTMDet checkpoint was not supplied as an exact path.
# Set this after confirming the actual best checkpoint.
RTMDET_WEIGHTS = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection"
    r"\runs\rtmdet\rtmdet_tiny_640_6gb\best_coco_bbox_mAP.pth"
)

# ============================================================
# RTMDet CONFIG
# ============================================================

RTMDET_CONFIG = Path(
    r"E:\rename\yolo_work_out\resrcher\marine_debris_detection"
    r"\"configs\rtmdet"
)

# ============================================================
# OUTPUTS
# ============================================================

RESULTS_H1 = PROJECT / "results" / "H1"
RESULTS_H2 = PROJECT / "results" / "H2"
RESULTS_H3 = PROJECT / "results" / "H3"
RESULTS_H4 = PROJECT / "results" / "H4"
RESULTS_STATISTICS = PROJECT / "results" / "statistics"
RESULTS_TABLES = PROJECT / "results" / "chapter4_tables"

CORRUPTION_DATA = PROJECT / "data" / "corruption"

DOMAIN_SHIFT_DATA = PROJECT / "data" / "domain_shift"

ADAPTATION_DATA = PROJECT / "data" / "adaptation"

# ============================================================
# H2 CORRUPTIONS
# ============================================================

CORRUPTIONS = {
    "brightness": {
        "type": "brightness",
        "severity": 2,
    },
    "contrast": {
        "type": "contrast",
        "severity": 2,
    },
    "gaussian_noise": {
        "type": "gaussian_noise",
        "severity": 2,
    },
    "gaussian_blur": {
        "type": "gaussian_blur",
        "severity": 2,
    },
    "motion_blur": {
        "type": "motion_blur",
        "severity": 2,
    },
    "jpeg_compression": {
        "type": "jpeg_compression",
        "severity": 2,
    },
}

# ============================================================
# H3 TARGET DOMAIN
# ============================================================

# Set this to the real Iranian coastal / UAV target-domain dataset
# when it is available.

TARGET_DOMAIN_ROOT = Path(
    r"E:\rename\yolo_work_out\resrcher\TARGET_DOMAIN_DATASET"
)

TARGET_IMAGES = TARGET_DOMAIN_ROOT / "images"
TARGET_LABELS = TARGET_DOMAIN_ROOT / "labels"

# ============================================================
# H4
# ============================================================

ADAPTATION_EPOCHS = 20

ADAPTATION_LR = 0.0001

RANDOM_SEED = 42