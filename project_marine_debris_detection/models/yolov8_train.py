# -*- coding: utf-8 -*-
"""
====================================================================================================
YOLOv8 / RT-DETR / RTMDet TRAINER (پروپوزال محور)
====================================================================================================
این کد بر اساس پروپوزال پایان‌نامه بهبود تشخیص زباله‌های دریایی با مقایسه سه مدل YOLOv8، RT-DETR و RTMDet
طراحی شده است. تغییرات اعمال شده:
- پشتیبانی از هر سه مدل با انتخاب نوع مدل
- تنظیم batch_size مؤثر = 32 با استفاده از Gradient Accumulation
- ذخیره چک‌پوینت هر epoch (save_period=1)
- نمایش نتایج اعتبارسنجی (با معیار mAP@0.5 و mAP@0.5:0.95) هر ۵ epoch
- هماهنگی با پروتکل آزمایشی پروپوزال
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
from ultralytics import YOLO

# فرض بر این است که config.py شامل TRAIN_CONFIG, CLASSES, RESULTS_ROOT است
from config import TRAIN_CONFIG, CLASSES, RESULTS_ROOT


class YOLOv8Trainer:
    def __init__(self, data_yaml, model_type="yolov8", model_size="n", effective_batch_size=32):
        """
        Args:
            data_yaml: مسیر فایل data.yaml
            model_type: نوع مدل ("yolov8", "rtdetr", "rtmdet")
            model_size: اندازه مدل (برای YOLOv8: n, s, m, l, x؛ برای RT-DETR: l, x؛ برای RTMDet: l, x)
            effective_batch_size: اندازه دسته مؤثر نهایی (هدف) - با استفاده از gradient accumulation به آن می‌رسیم
        """
        self.data_yaml = str(data_yaml)
        self.model_type = model_type.lower()
        self.model_size = model_size
        self.effective_batch_size = effective_batch_size
        # نام مدل برای ذخیره‌سازی
        self.model_name = f"{model_type}{model_size}"
        self.base_dir = RESULTS_ROOT / self.model_name
        self._create_dirs()
        self.history = defaultdict(list)

    def _create_dirs(self):
        """ایجاد پوشه‌های خروجی"""
        subdirs = ["checkpoints", "logs", "test_images", "heatmaps", "metrics"]
        for sub in subdirs:
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def _get_pretrained_weight(self):
        """دریافت نام وزن پیش‌آموزش‌دیده بر اساس نوع مدل"""
        if self.model_type == "yolov8":
            return f"yolov8{self.model_size}.pt"
        elif self.model_type == "rtdetr":
            # RT-DETR معمولاً با اندازه‌های l یا x ارائه می‌شود
            return f"rtdetr-{self.model_size}.pt" if self.model_size in ['l', 'x'] else "rtdetr-l.pt"
        elif self.model_type == "rtmdet":
            return f"rtmdet-{self.model_size}.pt" if self.model_size in ['l', 'x'] else "rtmdet-l.pt"
        else:
            raise ValueError(f"نوع مدل {self.model_type} پشتیبانی نمی‌شود. گزینه‌ها: yolov8, rtdetr, rtmdet")

    def _get_model_class(self):
        """برگرداندن نام کلاس مدل برای ذخیره‌سازی (اختیاری)"""
        return self.model_type.upper()

    def train(self):
        """آموزش مدل با تنظیمات پروپوزال"""
        print(f"\n{'='*60}")
        print(f"🚀 آموزش {self.model_name} ({self.model_type.upper()})")
        print(f"   اندازه دسته مؤثر: {self.effective_batch_size}")
        print(f"{'='*60}")

        # بارگذاری مدل با وزن پیش‌آموزش‌دیده
        weight = self._get_pretrained_weight()
        model = YOLO(weight)

        # محاسبه gradient accumulation
        # batch_size واقعی از TRAIN_CONFIG گرفته می‌شود
        real_batch = TRAIN_CONFIG.get("batch_size", 8)
        # اگر batch_size واقعی بزرگتر از هدف باشد، نمی‌توانیم با accumulation کاهش دهیم
        if real_batch > self.effective_batch_size:
            print(f"⚠️ هشدار: batch_size واقعی ({real_batch}) بزرگتر از هدف ({self.effective_batch_size}) است.")
            print("   از batch_size واقعی استفاده می‌شود و accumulation=1.")
            accum = 1
            used_batch = real_batch
        else:
            if self.effective_batch_size % real_batch != 0:
                print(f"⚠️ هشدار: {self.effective_batch_size} بر {real_batch} بخش‌پذیر نیست.")
                # نزدیک‌ترین عدد بخش‌پذیر را انتخاب می‌کنیم
                new_effective = (self.effective_batch_size // real_batch) * real_batch
                if new_effective == 0:
                    new_effective = real_batch
                print(f"   اندازه دسته مؤثر به {new_effective} تغییر یافت.")
                self.effective_batch_size = new_effective
            accum = self.effective_batch_size // real_batch
            used_batch = real_batch

        print(f"   batch_size واقعی: {used_batch}, accumulation steps: {accum}")

        # تعریف callback برای نمایش نتایج اعتبارسنجی هر ۵ epoch
        def on_epoch_end(trainer):
            epoch = trainer.epoch
            if epoch % 5 == 0:
                # دریافت متریک‌های اعتبارسنجی از trainer
                metrics = trainer.metrics
                if metrics:
                    mAP50 = metrics.get("mAP50", -1)
                    mAP50_95 = metrics.get("mAP50-95", -1)
                    print(f"\n📊 [Epoch {epoch}] Validation metrics:")
                    print(f"   mAP@0.5: {mAP50:.4f}")
                    print(f"   mAP@0.5:0.95: {mAP50_95:.4f}")
                else:
                    print(f"\n📊 [Epoch {epoch}] No validation metrics yet.")

        # دیکشنری callbacks
        callbacks = {
            "on_epoch_end": on_epoch_end
        }

        # شروع آموزش
        results = model.train(
            data=self.data_yaml,
            epochs=TRAIN_CONFIG.get("epochs", 150),
            batch=used_batch,                     # اندازه دسته واقعی
            accumulate=accum,                     # تعداد steps برای انباشت گرادیان
            imgsz=TRAIN_CONFIG.get("img_size", 640),
            lr0=TRAIN_CONFIG.get("lr", 0.0001),
            device=TRAIN_CONFIG.get("device", 0),
            workers=TRAIN_CONFIG.get("workers", 4),
            patience=TRAIN_CONFIG.get("patience", 50),
            project=str(self.base_dir / "checkpoints"),
            name="yolov8",
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=1,                        # ذخیره چک‌پوینت هر epoch
            plots=True,
            amp=False,                            # غیرفعال کردن AMP طبق پروپوزال
            callbacks=callbacks,                  # افزودن callback
        )

        # ذخیره تاریخچه
        self.history["train_loss"] = results.history.get("train/box_loss", [])
        self.history["val_loss"] = results.history.get("val/box_loss", [])
        self.history["cls_loss"] = results.history.get("train/cls_loss", [])
        self.history["mAP50"] = results.history.get("metrics/mAP50(B)", [])
        self.history["mAP50-95"] = results.history.get("metrics/mAP50-95(B)", [])
        self.history["precision"] = results.history.get("metrics/precision(B)", [])
        self.history["recall"] = results.history.get("metrics/recall(B)", [])

        # ذخیره تاریخچه در JSON
        hist_path = self.base_dir / "logs" / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # رسم نمودارها
        self._plot_history()

        print(f"\n✅ آموزش {self.model_name} کامل شد.")
        print(f"📁 نتایج در {self.base_dir}")

        return model, results

    def _plot_history(self):
        """نمودارهای آموزش (بدون تغییر)"""
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
        f1 = [2 * (p[i] * r[i]) / (p[i] + r[i] + 1e-6) for i in range(min(len(p), len(r)))]
        axes[1, 1].plot(f1, label="F1-Score")
        axes[1, 1].set_title("F1-Score")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1")
        axes[1, 1].legend()
        axes[1, 1].grid()

        plt.tight_layout()
        plt.savefig(self.base_dir / "logs" / "training_history.png", dpi=150)
        plt.close()

    def evaluate(self, model, test_dir, num_samples=20):
        """ارزیابی مدل روی تصاویر تست (بدون تغییر اساسی)"""
        from ultralytics import YOLO

        if isinstance(model, str):
            model = YOLO(model)

        print(f"\n📊 ارزیابی {self.model_name} روی داده‌های تست...")

        # ارزیابی با Ultralytics
        results = model.val(
            data=self.data_yaml,
            split="test",
            batch=TRAIN_CONFIG.get("batch_size", 8),
            imgsz=TRAIN_CONFIG.get("img_size", 640),
            device=TRAIN_CONFIG.get("device", 0),
            save_json=True,
            save_txt=True,
            save_conf=True,
            plots=True,
            project=str(self.base_dir / "checkpoints"),
            name="eval",
        )

        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "f1": results.box.f1,
        }

        with open(self.base_dir / "metrics" / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._save_test_images(model, test_dir, num_samples)
        self._generate_heatmaps(model, test_dir, num_samples=10)

        return metrics

    def _save_test_images(self, model, test_dir, num_samples=20):
        """ذخیره تصاویر تست با GT و Predict (بدون تغییر)"""
        from ultralytics import YOLO

        if isinstance(model, str):
            model = YOLO(model)

        image_files = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
        image_files = image_files[:num_samples]

        save_dir = self.base_dir / "test_images"
        save_dir.mkdir(exist_ok=True)

        for img_path in tqdm(image_files, desc=f"🎨 ذخیره تصاویر تست {self.model_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = model(img_path, conf=0.25, iou=0.45)
            annotated = results[0].plot() if results else img_rgb.copy()

            # افزودن GT
            gt_boxes = self._load_gt_boxes(img_path)
            for box in gt_boxes:
                x1, y1, x2, y2 = box["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"GT: {box['class']}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            save_path = save_dir / f"{img_path.stem}_pred.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    def _load_gt_boxes(self, img_path):
        """بارگذاری GT از فایل لیبل YOLO (بدون تغییر)"""
        label_file = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
        if not label_file.exists():
            label_file = Path(str(img_path).replace("images", "labels").replace(".png", ".txt"))

        boxes = []
        if label_file.exists():
            img = cv2.imread(str(img_path))
            if img is None:
                return boxes
            h, w = img.shape[:2]
            try:
                lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                return boxes

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                boxes.append({
                    "class": CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}",
                    "bbox": [x1, y1, x2, y2]
                })
        return boxes

    def _generate_heatmaps(self, model, test_dir, num_samples=10):
        """تولید Heatmap با Grad-CAM (بدون تغییر)"""
        try:
            import torch
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            print("⚠️ grad-cam نصب نیست. نصب کنید: pip install grad-cam")
            return

        from ultralytics import YOLO

        if isinstance(model, str):
            model = YOLO(model)

        image_files = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
        image_files = image_files[:num_samples]

        save_dir = self.base_dir / "heatmaps"
        save_dir.mkdir(exist_ok=True)

        try:
            target_layers = [model.model.model[-1].conv]
        except Exception:
            print("⚠️ یافتن لایه هدف برای Grad-CAM ناموفق بود")
            return

        for img_path in tqdm(image_files, desc=f"🔥 تولید Heatmap {self.model_name}"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

                from ultralytics.utils import ops
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
                img_tensor = ops.resize(img_tensor, (640, 640))

                cam = GradCAM(model=model.model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=img_tensor)

                cam_image = show_cam_on_image(img_rgb, grayscale_cam[0], use_rgb=True)
                save_path = save_dir / f"{img_path.stem}_heatmap.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"⚠️ خطا در تولید Heatmap برای {img_path.name}: {e}")
                continue