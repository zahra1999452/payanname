"""Training source package. Dedicated RT-DETR/RTMDet scripts have their own CLI.

Importing the package alone does not import GPU libraries or start training.
"""
from importlib import import_module

__all__ = ["UnifiedTrainer", "YOLOv8Trainer"]

def __getattr__(name):
    if name in __all__:
        module = import_module(".unified_trainer", __name__)
        return module.UnifiedTrainer
    raise AttributeError(name)
