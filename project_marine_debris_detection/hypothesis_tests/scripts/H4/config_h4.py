from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]

ADAPTATION_ROOT = (
    PROJECT /
    "data" /
    "adaptation"
)

TARGET_TRAIN = ADAPTATION_ROOT / "train"
TARGET_VAL = ADAPTATION_ROOT / "val"
TARGET_TEST = ADAPTATION_ROOT / "test"

MODEL_TO_ADAPT = "RTMDet"

EPOCHS = 20

LEARNING_RATE = 0.0001

IMAGE_SIZE = 640

SEED = 42