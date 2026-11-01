# Legacy wrapper: synthetic history/metrics removed; entry points disabled.
# -*- coding: utf-8 -*-
"""
====================================================================================================
RTMDet TRAINER (پروپوزال محور)
====================================================================================================
⚠️ این پیاده‌سازی برای RTMDet نیاز به MMDetection دارد:
   https://github.com/open-mmlab/mmdetection

تغییرات اعمال شده بر اساس پروپوزال:
- تنظیم batch_size مؤثر = 32 با استفاده از Gradient Accumulation
- ذخیره چک‌پوینت هر epoch (save_period=1)
- نمایش نتایج اعتبارسنجی (mAP@0.5 و mAP@0.5:0.95) هر ۵ epoch
- هماهنگی با پروتکل آزمایشی (آموزش ۱۵۰ epoch، بهینه‌ساز AdamW، نرخ یادگیری 0.0001)
====================================================================================================
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from config import TRAIN_CONFIG, CLASSES, RESULTS_ROOT


class RTMDetTrainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Legacy RTMDet wrapper is disabled: its former metrics were examples. "
            "Use train_rtmdet_tiny.py for real training/evaluation."
        )

    def _create_dirs(self):
        """ایجاد پوشه‌های خروجی"""
        subdirs = ["checkpoints", "logs", "test_images", "heatmaps", "metrics"]
        for sub in subdirs:
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def _generate_config_file(self):
        """
        تولید فایل کانفیگ RTMDet (فرمت MMDetection) بر اساس تنظیمات پروپوزال
        """
        # محاسبه batch_size واقعی و تعداد steps برای accumulation
        # در MMDetection معمولاً از batch_size=8 با 4 GPU استفاده می‌شود، اما ما با یک GPU کار می‌کنیم
        # برای رسیدن به effective 32، مثلاً batch_size=8 و accumulation=4
        real_batch = 8  # می‌توان از TRAIN_CONFIG.get("batch_size", 8) استفاده کرد
        accum = self.effective_batch_size // real_batch
        if accum < 1:
            accum = 1
            real_batch = self.effective_batch_size
        if self.effective_batch_size % real_batch != 0:
            # تنظیم مجدد به نزدیک‌ترین مقدار
            new_effective = (self.effective_batch_size // real_batch) * real_batch
            if new_effective == 0:
                new_effective = real_batch
            self.effective_batch_size = new_effective
            accum = self.effective_batch_size // real_batch

        # تعیین نام کانفیگ پایه بر اساس اندازه مدل
        size_map = {
            'tiny': 'rtmdet_tiny_8xb32-300e_coco.py',
            's': 'rtmdet_s_8xb32-300e_coco.py',
            'm': 'rtmdet_m_8xb32-300e_coco.py',
            'l': 'rtmdet_l_8xb32-300e_coco.py',
            'x': 'rtmdet_x_8xb32-300e_coco.py'
        }
        base_config = size_map.get(self.model_size, 'rtmdet_l_8xb32-300e_coco.py')

        config_content = f"""
# RTMDet Configuration for Marine Debris Detection
# Generated based on the proposal
# Base config: {base_config}

_base_ = ['configs/rtmdet/{base_config}']

# ==================== Data ====================
# مسیر داده‌ها (فرض بر این است که دیتاست به فرمت COCO آماده شده است)
dataset_type = 'CocoDataset'
data_root = '{os.path.dirname(self.data_yaml)}'

# ==================== Training settings ====================
# تعداد epoch ها طبق پروپوزال
train_cfg = dict(
    max_epochs={TRAIN_CONFIG.get('epochs', 150)},
    val_interval=5  # اعتبارسنجی هر ۵ epoch
)

# ==================== Optimizer (AdamW) ====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr={TRAIN_CONFIG.get('lr', 0.0001)}, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True
    ),
    # gradient accumulation برای رسیدن به effective batch size
    accumulative_counts={accum},
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

# ==================== Learning rate scheduler ====================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=30
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=30,
        end={TRAIN_CONFIG.get('epochs', 150)},
        T_max={TRAIN_CONFIG.get('epochs', 150) - 30},
        eta_min=0.000001
    )
]

# ==================== Batch size (real) ====================
train_dataloader = dict(
    batch_size={real_batch},
    num_workers={TRAIN_CONFIG.get('workers', 4)},
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            # Augmentations طبق پروپوزال
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomCrop', crop_size=({TRAIN_CONFIG.get('img_size', 640)}, {TRAIN_CONFIG.get('img_size', 640)})),
            dict(type='PhotoMetricDistortion'),  # شامل ColorJitter
            dict(type='RandomGaussianNoise', var_limit=(10.0, 50.0)),
            dict(type='MotionBlur'),  # شبیه‌سازی حرکت پهپاد (نیاز به ثبت در pipeline)
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size={real_batch},
    num_workers={TRAIN_CONFIG.get('workers', 4)},
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=({TRAIN_CONFIG.get('img_size', 640)}, {TRAIN_CONFIG.get('img_size', 640)})),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader  # مشابه val

# ==================== Evaluation metrics ====================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# ==================== Save checkpoints every epoch ====================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,          # ذخیره هر epoch
        max_keep_ckpts=3,    # حداکثر ۳ چک‌پوینت اخیر نگهداری شود
        save_best='auto',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='VisualizationHook', interval=5)  # نمایش نتایج هر ۵ epoch
)

# ==================== Device ====================
device = '{TRAIN_CONFIG.get('device', 'cuda')}'
"""
        config_dir = self.base_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "rtmdet_config.py"
        with open(config_path, 'w') as f:
            f.write(config_content)
        return config_path

    def train(self):
        raise NotImplementedError("Legacy demo disabled; use train_rtmdet_tiny.py.")

    def _plot_history(self):
        """نمودارهای آموزش"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(self.history.get("train_loss", []), label="Train Loss")
        axes[0, 0].plot(self.history.get("val_loss", []), label="Val Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid()

        # mAP
        axes[0, 1].plot(self.history.get("mAP50", []), label="mAP@50")
        axes[0, 1].plot(self.history.get("mAP50-95", []), label="mAP@50-95")
        axes[0, 1].set_title("mAP")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mAP")
        axes[0, 1].legend()
        axes[0, 1].grid()

        # Precision & Recall
        axes[1, 0].plot(self.history.get("precision", []), label="Precision")
        axes[1, 0].plot(self.history.get("recall", []), label="Recall")
        axes[1, 0].set_title("Precision & Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid()

        # F1-Score
        p = self.history.get("precision", [])
        r = self.history.get("recall", [])
        if p and r:
            min_len = min(len(p), len(r))
            f1 = [2 * (p[i] * r[i]) / (p[i] + r[i] + 1e-6) for i in range(min_len)]
            axes[1, 1].plot(f1, label="F1-Score")
            axes[1, 1].set_title("F1-Score")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("F1")
            axes[1, 1].legend()
            axes[1, 1].grid()

        plt.tight_layout()
        plt.savefig(self.base_dir / "logs" / "training_history.png", dpi=150)
        plt.close()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Legacy demo disabled; use train_rtmdet_tiny.py --test.")

    def _save_test_images(self, model_path, test_dir, num_samples=20):
        """ذخیره تصاویر تست با پیش‌بینی (در صورت وجود مدل)"""
        print("⚠️ برای ذخیره تصاویر تست RTMDet، مدل باید بارگذاری شود.")
        print("   این قابلیت در صورت وجود فایل مدل و استفاده از MMDetection قابل اجراست.")
        save_dir = self.base_dir / "test_images"
        save_dir.mkdir(exist_ok=True)

        image_files = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
        image_files = image_files[:num_samples]

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                save_path = save_dir / f"{img_path.stem}_pred.jpg"
                cv2.imwrite(str(save_path), img)

        print(f"✅ {len(image_files)} تصویر تست در {save_dir} ذخیره شد.")