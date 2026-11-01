"""Small, dependency-free helpers for the audited training entry points."""
import csv
import importlib.util
import math
import os
from pathlib import Path

HISTORY_COLUMNS = {
    'train_loss': 'train/box_loss', 'val_loss': 'val/box_loss',
    'mAP50': 'metrics/mAP50(B)', 'mAP50-95': 'metrics/mAP50-95(B)',
    'precision': 'metrics/precision(B)', 'recall': 'metrics/recall(B)',
}

def read_training_history(csv_path):
    """Fail explicitly on absent, empty, malformed or incompatible epoch logs."""
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        rows = [{k.strip(): v.strip() for k, v in row.items() if k is not None}
                for row in reader]
    if not rows:
        raise ValueError(f'No completed epoch records in {csv_path}')
    missing = set(HISTORY_COLUMNS.values()) - rows[0].keys()
    if missing:
        raise ValueError(f'Missing Ultralytics CSV columns: {sorted(missing)}')
    result = {key: [float(row[column]) for row in rows]
              for key, column in HISTORY_COLUMNS.items()}
    if not all(math.isfinite(value) for values in result.values() for value in values):
        raise ValueError('Training CSV contains non-finite values')
    return result

def json_native(value):
    """Convert NumPy scalars/arrays recursively without requiring NumPy itself."""
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_native(value.tolist())
    if hasattr(value, 'item'):
        return json_native(value.item())
    return value

def discover_rtmdet_config():
    """Discover a config tree; its _base_ dependencies must remain adjacent."""
    override = os.environ.get('MMDET_RTMDET_CONFIG')
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f'MMDET_RTMDET_CONFIG does not exist: {candidate}')
        return candidate
    bundled = Path(__file__).resolve().parent/'config_snapshot/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    if bundled.is_file():
        return bundled
    spec = importlib.util.find_spec('mmdet')
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError('Install MMDetection or set MMDET_RTMDET_CONFIG')
    name = 'rtmdet_tiny_8xb32-300e_coco.py'
    candidates = []
    for root in spec.submodule_search_locations:
        candidates += [Path(root)/'.mim/configs/rtmdet'/name,
                       Path(root).parent/'configs/rtmdet'/name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError('MMDetection config tree missing; set MMDET_RTMDET_CONFIG with its _base_ files')
