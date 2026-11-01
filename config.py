"""Portable configuration added for this revision, not a recovered historic export.

Original full training configuration/environment is unavailable. Defaults below
follow explicit defaults in the archived YOLO trainer and do not prove its recipe.
Set MARINE_DATASET_ROOT to the recovered 13-class dataset before training.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("MARINE_DATASET_ROOT", str(PROJECT_ROOT / "data")))
RESULTS_ROOT = Path(os.environ.get("MARINE_RESULTS_ROOT", str(PROJECT_ROOT / "runs")))
CLASSES = ['animal_crab', 'animal_eel', 'animal_etc', 'animal_fish', 'animal_shells', 'animal_starfish', 'plant', 'trash_branch', 'trash_clothing', 'trash_net', 'trash_unknown_instance', 'trash_wreckage', 'trash_plastic']
TRAIN_CONFIG = {
    "epochs": 150, "batch_size": 8, "img_size": 640, "lr": 0.0001,
    "device": 0, "workers": 4, "patience": 50,
}
