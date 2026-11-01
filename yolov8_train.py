"""Compatibility name for the supported YOLO-only trainer; no parallel implementation."""
if __package__:
    from .unified_trainer import UnifiedTrainer
else:
    from unified_trainer import UnifiedTrainer

YOLOv8Trainer = UnifiedTrainer
__all__ = ["YOLOv8Trainer"]
