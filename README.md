# Marine Debris Detection — Training + Evaluation Pipeline

*[English version below the Persian section.](#english)*

---

## فارسی

مخزن کد پایان‌نامهٔ کارشناسی‌ارشد «تشخیص پسماندهای دریایی با یادگیری عمیق»؛ آموزش و مقایسهٔ سه
مدل **YOLOv8n**، **RT-DETR-L** و **RTMDet-Tiny** روی یک مجموعه‌دادهٔ ۱۳‌کلاسه، و آزمون چهار فرضیهٔ
پژوهشی (H1–H4) روی خروجی آن‌ها.

مخزن از سه بخش اصلی تشکیل شده: `models/` (آموزش سه مدل)، `hypothesis_tests/` (آزمون فرضیه‌ها روی
چک‌پوینت‌های نهایی) و `docs/` (فایل پایان‌نامه).

**فهرست مطالب:**
[ساختار مخزن](#ساختار-مخزن) ·
[آموزش مدل‌ها](#آموزش-مدل‌ها-models) ·
[فرمت داده هر مدل](#فرمت-داده-هر-مدل) ·
[آزمون فرضیه‌ها](#آزمون-فرضیه‌ها-hypothesis_tests) ·
[پیش‌نیازها](#پیش‌نیازها) ·
[وضعیت مخزن و نکات مهم](#وضعیت-این-مخزن-و-نکات-لازم-برای-خواننده) ·
[English ⤵](#english)

### ساختار مخزن

```text
.
├── README.md
├── docs/
│   └── پایان_نامه.docx                # متن کامل پایان‌نامه
│
├── models/                            # آموزش سه مدل
│   ├── __init__.py
│   ├── unified_trainer.py             # UnifiedTrainer — مسیر واقعیِ آموزش YOLOv8n (به‌کار رفته)
│   ├── yolov8_train.py                # YOLOv8Trainer — نسخهٔ تقریباً یکسان/موازیِ UnifiedTrainer
│   ├── rtdetr_train.py                # پیش‌نویس RT-DETR (batch مؤثر=16) — استفاده نشد
│   ├── run_rtdetr.py                  # پیش‌نویس سادهٔ RT-DETR (batch=8) — استفاده نشد
│   ├── train_rtdetr_l_6gb.py          # اسکریپت واقعیِ آموزش RT-DETR-L (به‌کار رفته)
│   ├── rtmdet_train.py                # پیش‌نویس/wrapper مولد کانفیگ RTMDet — نقشی جانبی
│   ├── train_rtmdet_tiny.py           # اسکریپت واقعیِ آموزش RTMDet-Tiny (به‌کار رفته)
│   ├── check_rtdetr.py                # تشخیص محیط (GPU/CUDA/PyTorch) — ابزار جانبی
│   └── test.py                        # ابزار جانبی: تبدیل فرمول‌های LaTeX نسخهٔ قدیمی پایان‌نامه به یونیکد
│
└── hypothesis_tests/                  # اجرای آزمون فرضیه‌ها H1 تا H4
    ├── README.md / README.txt         # توضیح مختصر همین پوشه
    ├── requirements_statistics.txt
    ├── plot_all_results.py
    ├── .gitignore
    │
    ├── configs/
    │   └── config.py                  # تنظیمات سراسری (TRAIN_CONFIG، CLASSES، مسیرها)
    │
    ├── H1_145/                        # آرشیو اجرای مرجع H1 (epoch 146 پایه)
    ├── logs/
    │
    ├── scripts/
    │   ├── common/                    # توابع مشترک (IoU، matching، I/O)
    │   │   ├── audit_test_split.py
    │   │   └── io_utils.py
    │   ├── H1/
    │   │   ├── H1_rerun.py
    │   │   ├── run_h1_complete.py
    │   │   ├── run_h1_statistics.py
    │   │   └── README.md
    │   ├── H2/
    │   │   ├── H2_robustness_cached.py
    │   │   ├── make_corruptions.py
    │   │   ├── run_h2_complete.py
    │   │   ├── run_h2_statistics.py
    │   │   └── README.md
    │   ├── H3/
    │   │   ├── build_H3_target_dataset.py
    │   │   ├── check_target_domain.py
    │   │   ├── H3_domain_shift_complete.py
    │   │   ├── make_h3_domain_shift.py
    │   │   ├── run_h3_all_conditions.py
    │   │   ├── run_h3_complete.py
    │   │   ├── run_h3_statistics.py
    │   │   └── README.md
    │   ├── H4/
    │   │   ├── config_h4.py
    │   │   ├── run_h4_complete.py
    │   │   ├── run_h4_end_to_end.py
    │   │   ├── run_h4_statistics.py
    │   │   └── README.md
    │   ├── make_chapter4_tables.py     # تولید جدول‌های نهایی فصل ۴ از نتایج خام
    │   └── select_h4_model.py          # قاعدهٔ انتخاب مدل ورودی به H4 بر اساس رتبهٔ H1–H3
    │
    ├── results2/                       # خروجی نهایی گزارش‌شده (نسخهٔ resumable/cached)
    │   ├── H1/cache/
    │   ├── H2/cache/
    │   ├── H3/target_dataset/{images,labels}/test/   # تصاویر/لیبل‌های واقعی ایگنور شده‌اند؛ فقط
    │   │                                              # .gitkeep + manifest.csv ردیابی می‌شوند
    │   ├── comparison/
    │   ├── plots1/{H1,H2,H3}/
    │   └── MASTER_PLOTS/                              # ALL_METRICS_INVENTORY.csv (~296MB) ایگنور
    │                                                   # شده — محدودیت آپلود گیت‌هاب؛ نکات را ببینید
    │
    └── runs/
        └── H4/yolov8_synthetic_adaptation/weights/   # best.pt مدل سازگارشدهٔ H4

models/runs/detect/results/RT-DETR/     # خروجی اجراهای RT-DETR (کنار models/، نه زیرشاخهٔ آن؛
├── rtdetr-l-6gb/weights/               #  در این زیپ به دلیل حجم زیاد شامل نشده — بخش «نکات» را ببینید)
├── rtdetr-r18/weights/                 # آزمایش رهاشدهٔ اولیه (backbone سبک‌تر R18) — گزارش نشده
└── rtdetr-r18-6gb/weights/             # آزمایش رهاشدهٔ اولیه — گزارش نشده
```

> **نکته دربارهٔ این نسخهٔ فشرده‌شده:** فایل `hypothesis_tests/runs/H4.zip` (آرشیو چک‌پوینت‌های
> epoch‌به‌epoch مدل YOLOv8n سازگارشده، حدود ۳۱۱ مگابایت) طبق تصمیم مشترک از این بسته حذف شده
> چون شامل فایل‌های وزن مدل (`*.pt`) است — دقیقاً همان نوع فایلی که `.gitignore` خودِ پروژه هم
> قصد نادیده‌گرفتنش را دارد (`results/**/*.pt`, `results/**/*.pth`). پوشهٔ `models/runs/` هم در
> زیپ اصلی وجود نداشت و صرفاً برای مستندسازی مسیر خروجی در این README ذکر شده است. اگر
> چک‌پوینت‌های نهایی لازم است، باید جداگانه (خارج از ریپو، مثلاً با Git LFS یا فضای ابری) نگه‌داری
> شوند.
>
> به همین ترتیب، تصاویر و لیبل‌های واقعیِ
> `hypothesis_tests/results2/H3/target_dataset/{images,labels}/test/` (حدود ۲٬۴۶۲ تصویر JPG +
> ۲٬۴۶۲ لیبل YOLO `.txt`، در مجموع ~۱۰۷ مگابایت) هم از این بسته حذف شده‌اند؛ این مسیر در
> `hypothesis_tests/.gitignore` اضافه شده تا داده‌های حجیم دیتاست هدف اصلاً وارد گیت نشوند، اما
> **ساختار پوشه حفظ شده**: هر دو پوشهٔ `images/test/` و `labels/test/` یک فایل نشانگر `.gitkeep`
> دارند و `manifest.csv` (که پارامترهای تولید هر تصویر synthetic — brightness، contrast، color
> gain، haze، blur، exposure، saturation، noise، rotation، jpeg quality و غیره — را نگه می‌دارد)
> کامل و دست‌نخورده باقی مانده است. برای اجرای واقعی H3، این دو پوشه باید دوباره با
> `scripts/H3/build_H3_target_dataset.py` پر شوند.
>
> در همین راستا، فایل `hypothesis_tests/results2/MASTER_PLOTS/ALL_METRICS_INVENTORY.csv` هم به
> **دلیل محدودیت آپلود گیت‌هاب** (سقف سخت ۱۰۰ مگابایت به‌ازای هر فایل، با هشدار از ۵۰ مگابایت به
> بالا) از `.gitignore` و از این بسته حذف شده است. این فایل حدود **۲۹۶ مگابایت** و نزدیک به
> ۱٫۹۴ میلیون ردیف بود؛ از پوشهٔ `MASTER_PLOTS/` به‌دست آمده و یک inventory تخت (long-format) از
> تمام متریک‌های per-image (Precision/Recall/F1/…) بود که از فایل‌های `per_image_metrics.csv`
> هر چهار آزمایش — یعنی `results2/H1/`، `results2/H2/`، `results2/H3/` و `results2/H4/` (ستون
> `file` در خودِ فایل به مسیر `per_image_metrics.csv` هرکدام اشاره می‌کرد) — کنار هم جمع شده بود.
> هیچ اسکریپتی برای تولید این فایل در مخزن موجود نیست (احتمالاً دستی یا با اسکریپتی خارج از این
> مخزن ساخته شده)، بنابراین اگر دوباره لازم شد باید از نو از همان چهار فایل per_image_metrics.csv
> بازسازی شود. نسخهٔ خلاصه‌شدهٔ آن، `METRIC_SUMMARY_ALL_HYPOTHESES.csv` (کمتر از ۳ کیلوبایت)، در
> همان پوشه و در گیت باقی مانده است.

### آموزش مدل‌ها (`models/`)

هر سه مدل به یک شکل آموزش داده نشده‌اند؛ مسیر واقعیِ هرکدام متفاوت است:

**YOLOv8n — از طریق `UnifiedTrainer` (`unified_trainer.py`)**

تنها مسیری که برای هر سه مدل طراحی شده بود ولی واقعاً **فقط برای YOLOv8n** به‌کار رفت.
`UnifiedTrainer.train()` برای `model_type="yolov8"` مستقیماً `ultralytics.YOLO.train()` را با
تنظیمات زیر صدا می‌زند:

- `epochs=150`، `imgsz=640`، `lr0=0.0001`، `save_period=1` (چک‌پوینت هر epoch)
- batch واقعی از `TRAIN_CONFIG`، با `nbs` برای شبیه‌سازی batch مؤثر بزرگ‌تر
- **Resume خودکار**: `_find_last_checkpoint()` آخرین `.pt` را در
  `checkpoints/yolov8/weights/` پیدا و آموزش را از همان‌جا ادامه می‌دهد

فایل تقریباً هم‌ارز `yolov8_train.py` (کلاس `YOLOv8Trainer`، همان ساختار متدها) هم در مخزن هست؛
همان کد است با نام دیگر و در عمل استفاده نشد — export رسمی پکیج (`__init__.py`) هم از همین فایل
است، اما نسخهٔ اجراشده، `unified_trainer.py` بوده.

> ⚠️ **نکتهٔ فنی مهم دربارهٔ همین فایل** (تأییدشده با بازبینی کد): شاخهٔ `else` در `train()` و
> `evaluate()` — یعنی زمانی که `model_type` برابر `"rtdetr"` یا `"rtmdet"` باشد — هیچ آموزش یا
> ارزیابی واقعی انجام نمی‌دهد. فقط یک پیام چاپ می‌کند و سپس یک تاریخچهٔ **ساختگی** با فرمول
> `math.exp` و متریک‌های **hardcode**‌شده (`mAP50=0.48`/`0.52`, `FPS=25`/`32` و…) در خروجی
> می‌نویسد. این شاخه هرگز برای RT-DETR-L یا RTMDet-Tiny واقعی اجرا نشده، چون آن دو مدل از
> اسکریپت‌های کاملاً جداگانهٔ زیر آموزش دیده‌اند؛ اما چون امضای تابع عمومی است
> (`model_type="rtdetr"` یا `"rtmdet"`)، اگر کسی این کلاس را برای آن دو مدل صدا بزند، بی‌آنکه
> خطا بگیرد اعداد ساختگی تحویل می‌گیرد. توصیه می‌شود این شاخه یا حذف شود یا به‌صراحت
> `NotImplementedError` بزند.

**RT-DETR-L — از طریق `train_rtdetr_l_6gb.py`**

اسکریپت مستقل (بدون استفاده از کلاس‌های بالا) که چک‌پوینت نهاییِ گزارش‌شده در پایان‌نامه را
تولید کرده؛ نام اجرا `rtdetr_l_640_6gb_diskcache`، منطبق با پوشهٔ خروجی `rtdetr-l-6gb/`.
ویژگی‌ها: batch=6 (به‌جای مقادیر بزرگ‌تر در پیش‌نویس‌ها)، AMP فعال، Disk Cache، Resume واقعی از
`last.pt` با حفظ optimizer/scheduler/epoch state.

دو فایل دیگر — `rtdetr_train.py` (تابع `train_rtdetr()`, batch مؤثر=16 با accumulation) و
`run_rtdetr.py` (ساده‌ترین نسخه, batch=8) — پیش‌نویس‌های اولیه‌اند و در تولید نتیجهٔ نهایی نقشی
نداشتند. پوشه‌های `rtdetr-r18/` و `rtdetr-r18-6gb/` هم مربوط به آزمایش‌های رهاشدهٔ اولیه با
backbone سبک‌تر R18 هستند و در نتایج نهایی (که فقط RT-DETR-**L** است) به‌کار نرفته‌اند.

**RTMDet-Tiny — از طریق `train_rtmdet_tiny.py`**

اسکریپت مستقل مبتنی بر MMEngine (`Runner`, `Config`, `Hook`) که مستقیماً روی چارچوب MMDetection
3.x کار می‌کند و از هیچ‌کدام از کلاس‌های بالا استفاده نمی‌کند: مکانیزم Resume اصلاح‌شدهٔ MMEngine
(`resume=True` + `load_from=<checkpoint>`)، رفع ناسازگاری `CSPNeXtPAFPN` با حذف
`deepen_factor`/`widen_factor`، checkpoint و mAP اعتبارسنجی هر epoch، AMP فعال، Gradient
accumulation=4، batch=4، اندازهٔ ورودی 640.

فایل `rtmdet_train.py` (کلاس `RTMDetTrainer`) یک تولیدکنندهٔ کانفیگ MMDetection بر پایهٔ
`TRAIN_CONFIG` است، نه مسیر آموزش نهایی؛ نقش جانبی/کمکی دارد.

**ابزار جانبی**

`check_rtdetr.py` هیچ نقشی در آموزش ندارد؛ صرفاً محیط اجرا (نسخهٔ PyTorch، در دسترس‌بودن CUDA،
مشخصات GPU و VRAM آزاد) را برای عیب‌یابی چاپ می‌کند. `test.py` هم کد آموزش نیست؛ ابزاری برای
تبدیل فرمول‌های LaTeX خام یک نسخهٔ قدیمی فایل پایان‌نامه به یونیکد است.

### فرمت داده هر مدل

هر سه مدل روی یک taxonomy مشترک (۱۳ کلاس) کار می‌کنند، اما فرمت ورودی‌شان یکسان نیست:

| | YOLOv8n | RT-DETR-L | RTMDet-Tiny |
|---|---|---|---|
| فریم‌ورک بارگذاری | Ultralytics | Ultralytics | MMDetection |
| فرمت annotation | YOLO `.txt` | YOLO `.txt` (همان `data.yaml`) | COCO JSON |
| مختصات جعبه | نرمال‌شده [0,1]، مرکز جعبه | نرمال‌شده [0,1]، مرکز جعبه | پیکسلی مطلق، گوشهٔ بالا-چپ |
| شناسهٔ کلاس | از 0 | از 0 | از 1 (قرارداد COCO) |

**YOLOv8n و RT-DETR-L** چون هر دو از طریق `ultralytics.YOLO` بارگذاری می‌شوند، دقیقاً همان
data loader و همان `data.yaml` را به اشتراک می‌گذارند:

```
class_id  x_center  y_center  width  height    # هر مقدار در [0,1]
```

**RTMDet-Tiny** — MMDetection به‌صورت پیش‌فرض COCO می‌خواهد: `bbox: [x_min, y_min, w, h]` با
مقادیر پیکسلی مطلق و `category_id` از ۱. در این پروژه، دیتاست اصلی به فرمت YOLO نگهداری می‌شود
و پیش از آموزش RTMDet-Tiny به COCO تبدیل شده است (طبق `rtmdet_train.py`).

> معماری هر سه مدل (Backbone/Neck/Head) در متن پایان‌نامه (پوشهٔ `docs/`) به‌تفصیل آمده و اینجا
> تکرار نشده است.

سیزده کلاسِ taxonomy نهایی:

```
animal_crab, animal_eel, animal_etc, animal_fish, animal_shells, animal_starfish, plant,
trash_branch, trash_clothing, trash_net, trash_unknown_instance, trash_wreckage, trash_plastic
```

### آزمون فرضیه‌ها (`hypothesis_tests/`)

| فرضیه | موضوع | خروجی اصلی |
|---|---|---|
| **H1** | مقایسهٔ عملکرد پایه سه مدل روی مجموعه آزمون مشترک | Friedman، Wilcoxon زوجی + اصلاح Holm، Kendall's W |
| **H2** | پایداری در برابر ۶ اغتشاش کنترل‌شده تصویری (brightness، contrast، gaussian_noise، gaussian_blur، motion_blur، jpeg_compression) | افت نسبی (Relative Drop) و درصد حفظ عملکرد (Retention) به ازای هر مدل/شرط |
| **H3** | تغییر دامنه (Domain Shift) — یک نسخهٔ synthetic/controlled و یک شرط واقعی (`uav_coastal_domain_shift`) | میانگین Degradation به ازای هر مدل و شرط |
| **H4** | سازگارسازی محدود دامنه (Domain Adaptation) روی مدل منتخب (YOLOv8n) | مقایسهٔ paired قبل/بعد از adaptation (Wilcoxon signed-rank، bootstrap CI) |

اسکریپت‌های H2 و H3 به‌صورت **resumable/cached** نوشته شده‌اند: نتایج inference هر تصویر در
`results2/<HX>/cache/` ذخیره می‌شود و اجرای مجدد بدون پرچم `--force` فقط تصاویر باقی‌مانده را
پردازش می‌کند.

**معیارهای ارزیابی**

- تطبیق پیش‌بینی/GT: class-aware، آستانهٔ IoU = 0.50، آستانهٔ Confidence = 0.25
- معیار اصلی: F1-score در سطح هر تصویر (Precision/Recall/F1 بر مبنای TP/FP/FN)
- معیار انتخاب checkpoint: بیشترین mAP@0.50:0.95 روی Validation
- آزمون‌های آماری: Friedman + Wilcoxon signed-rank (اصلاح Holm) برای H1، Wilcoxon signed-rank
  برای H4، rank-biserial correlation به‌عنوان اندازهٔ اثر در هر دو

### پیش‌نیازها

```text
python >= 3.10
ultralytics          # YOLOv8n و RT-DETR-L (models/, hypothesis_tests/)
mmdetection           # RTMDet-Tiny (models/train_rtmdet_tiny.py)
mmengine
numpy, pandas, scipy, statsmodels
opencv-python
matplotlib
tqdm
```

نصب پکیج‌های آماری موردنیاز `hypothesis_tests/`:

```bash
pip install -r hypothesis_tests/requirements_statistics.txt
```

> YOLOv8n و RT-DETR-L از طریق Ultralytics و RTMDet-Tiny از طریق MMDetection/MMEngine بارگذاری
> می‌شوند؛ این دو محیط معمولاً باید جدا نصب شوند (نسخه‌های PyTorch/CUDA متفاوت رایج است).

### وضعیت این مخزن و نکات لازم برای خواننده

- چک‌پوینت‌های نهاییِ گزارش‌شده در پایان‌نامه: YOLOv8n از `unified_trainer.py`، RT-DETR-L از
  `train_rtdetr_l_6gb.py` (پوشهٔ `rtdetr-l-6gb`)، RTMDet-Tiny از `train_rtmdet_tiny.py`.
- فایل‌های `rtdetr_train.py`، `run_rtdetr.py`، `rtmdet_train.py` (کانفیگ‌ساز) و پوشه‌های
  `rtdetr-r18*` پیش‌نویس/آزمایش‌های اولیه‌اند و در نتیجهٔ نهایی نقشی ندارند.
- شاخهٔ RT-DETR/RTMDet در `unified_trainer.py`/`yolov8_train.py` (متد `train`/`evaluate` برای
  `model_type != "yolov8"`) داده‌های **ساختگی** برمی‌گرداند و نباید با نتایج واقعی گزارش‌شده در
  پایان‌نامه اشتباه گرفته شود (به بخش YOLOv8n بالا نگاه کنید).
- مقادیر گزارش‌شده در فصل ۴ پایان‌نامه از خروجی `hypothesis_tests/results2/` استخراج شده‌اند.
- مسیر دیتاست اصلی به‌صورت مطلق و محلی در `hypothesis_tests/configs/config.py` و
  `README.txt` ثبت شده (روی یک درایو ویندوزی خاص)؛ پیش از اجرای مجدد باید در محیط جدید
  بازتعریف شود.
- فایل‌های حجیم (چک‌پوینت‌های `.pt` در `runs/H4.zip`) از این بستهٔ فشرده‌شده حذف شده‌اند تا حجم
  تحویل معقول بماند — به یادداشت بالای بخش «ساختار مخزن» مراجعه کنید.
- تصاویر/لیبل‌های واقعیِ `results2/H3/target_dataset/{images,labels}/test/` هم در `.gitignore`
  مربوطه ایگنور شده‌اند و از این بسته حذف شده‌اند؛ فقط ساختار پوشه (با `.gitkeep`) و `manifest.csv`
  باقی مانده — همان یادداشت بالا را ببینید.
- `results2/MASTER_PLOTS/ALL_METRICS_INVENTORY.csv` (~۲۹۶ مگابایت، فراتر از سقف ۱۰۰ مگابایتیِ
  آپلود گیت‌هاب) هم ایگنور و از این بسته حذف شده؛ این فایل inventory تجمیعی متریک‌های
  per-image هر چهار آزمایش H1–H4 بود که از `MASTER_PLOTS/` به‌دست آمد — همان یادداشت بالا را
  ببینید. `METRIC_SUMMARY_ALL_HYPOTHESES.csv` (نسخهٔ خلاصه) دست‌نخورده در گیت مانده است.
- در `models/rtdetr_train.py` و `models/train_rtdetr_l_6gb.py`، بلوک‌های بزرگ کد کاملاً
  کامنت‌شده (پیش‌نویس‌های قدیمی‌تر همان اسکریپت، رویهم‌رفته حدود ۱۱۰۰ خط) از ابتدای فایل حذف
  شدند؛ منطق و خروجی کد فعال هیچ تغییری نکرده، فقط یک یادداشت کوتاه جایگزین آن‌ها شده است.

---

## English

Master's thesis code repository — *"Marine Debris Detection with Deep Learning."* Trains and
compares three models, **YOLOv8n**, **RT-DETR-L**, and **RTMDet-Tiny**, on a 13-class dataset,
and runs four research hypothesis tests (H1–H4) on their outputs.

The repository has three main parts: `models/` (training the three models), `hypothesis_tests/`
(hypothesis testing on the final checkpoints), and `docs/` (the thesis document).

See the directory tree in the Persian section above, and the per-hypothesis `README.md` files
under `hypothesis_tests/scripts/H1..H4/`, for the full layout (kept in Persian, the language of
the underlying research, to stay consistent across the repo).

**Contents:**
[Model training](#model-training-models) ·
[Data format per model](#data-format-per-model) ·
[Hypothesis testing](#hypothesis-testing-hypothesis_tests) ·
[Requirements](#requirements) ·
[Notes for readers](#notes-for-readers-of-this-repository) ·
[بازگشت به فارسی ⤴](#فارسی)

### Model training (`models/`)

The three models were **not** trained through a single unified path, despite `unified_trainer.py`
being written to support all three:

- **YOLOv8n** — actually trained via `UnifiedTrainer` (`unified_trainer.py`,
  `model_type="yolov8"`): wraps `ultralytics.YOLO.train()` with `epochs=150`, `imgsz=640`,
  `lr0=0.0001`, per-epoch checkpointing, and automatic resume from the last `.pt` file.
  `yolov8_train.py` (`YOLOv8Trainer`) is a near-duplicate that was **not** the path actually run.
- **RT-DETR-L** — trained via the standalone `train_rtdetr_l_6gb.py` (batch=6, AMP, disk cache,
  real resume with optimizer/scheduler/epoch state). `rtdetr_train.py` and `run_rtdetr.py` are
  earlier drafts not used for the reported checkpoint; the `rtdetr-r18*` output folders are
  abandoned lighter-backbone experiments, also not part of the reported results.
- **RTMDet-Tiny** — trained via the standalone `train_rtmdet_tiny.py`, built directly on
  MMDetection 3.x / MMEngine (fixed resume mechanism, `CSPNeXtPAFPN` compatibility fix, AMP,
  gradient accumulation=4, batch=4). `rtmdet_train.py` is only a config generator, not the
  training path itself.

> **Verified caveat:** in `unified_trainer.py` / `yolov8_train.py`, the code path for
> `model_type in {"rtdetr", "rtmdet"}` inside `train()`/`evaluate()` does **not** run real
> training or evaluation — it prints a message and returns a synthetic history (`math.exp`
> curves) and hardcoded metrics (e.g. `mAP50=0.48/0.52`, `FPS=25/32`). This path was never
> actually invoked for the real RT-DETR-L/RTMDet-Tiny runs (those used the separate scripts
> above), but it remains callable with that signature, so calling `UnifiedTrainer` directly for
> those two model types would silently return fake numbers. Removing this branch or making it
> raise `NotImplementedError` is recommended.

`check_rtdetr.py` is an environment-diagnostic utility (GPU/CUDA/VRAM), not part of training.
`test.py` is unrelated to training — it's a utility that converts raw LaTeX formulas from an
older draft of the thesis document into Unicode.

### Data format per model

All three models share the same 13-class taxonomy but expect different input formats:

| | YOLOv8n | RT-DETR-L | RTMDet-Tiny |
|---|---|---|---|
| Loaded via | Ultralytics | Ultralytics | MMDetection |
| Annotation format | YOLO `.txt` | YOLO `.txt` (same `data.yaml`) | COCO JSON |
| Box coordinates | normalized [0,1], center-based | normalized [0,1], center-based | absolute pixels, top-left corner |
| Class id | 0-indexed | 0-indexed | 1-indexed (COCO convention) |

The source dataset is kept in YOLO format and converted to COCO before RTMDet-Tiny training
(see `rtmdet_train.py`).

### Hypothesis testing (`hypothesis_tests/`)

| Hypothesis | Topic | Main output |
|---|---|---|
| **H1** | Baseline performance difference across the three models on the shared test set | Friedman, pairwise Wilcoxon + Holm correction, Kendall's W |
| **H2** | Robustness to 6 controlled image corruptions (brightness, contrast, gaussian noise, gaussian blur, motion blur, JPEG compression) | Relative drop and retention % per model/condition |
| **H3** | Domain shift — one synthetic/controlled variant and one real condition (`uav_coastal_domain_shift`) | Mean degradation per model/condition |
| **H4** | Limited domain adaptation on the selected model (YOLOv8n) | Paired before/after comparison (Wilcoxon signed-rank, bootstrap CI) |

H2 and H3 scripts are **resumable/cached**: per-image inference results are stored under
`results2/<HX>/cache/`, and re-running without `--force` only processes the remaining images.

Evaluation criteria: class-aware prediction/GT matching, IoU threshold = 0.50, confidence
threshold = 0.25; primary metric is per-image F1-score; checkpoint selection uses best
mAP@0.50:0.95 on validation.

### Requirements

```text
python >= 3.10
ultralytics          # YOLOv8n and RT-DETR-L
mmdetection           # RTMDet-Tiny
mmengine
numpy, pandas, scipy, statsmodels
opencv-python
matplotlib
tqdm
```

```bash
pip install -r hypothesis_tests/requirements_statistics.txt
```

Ultralytics and MMDetection/MMEngine usually need separate environments (they commonly pin
different PyTorch/CUDA versions).

### Notes for readers of this repository

- Reported final checkpoints: YOLOv8n from `unified_trainer.py`, RT-DETR-L from
  `train_rtdetr_l_6gb.py` (`rtdetr-l-6gb/`), RTMDet-Tiny from `train_rtmdet_tiny.py`.
- `rtdetr_train.py`, `run_rtdetr.py`, `rtmdet_train.py`, and the `rtdetr-r18*` output folders are
  early drafts/abandoned experiments with no role in the final reported results.
- The RT-DETR/RTMDet branch in `unified_trainer.py`/`yolov8_train.py` returns **synthetic** data
  and must not be confused with the real results reported in the thesis.
- Chapter 4 figures come from `hypothesis_tests/results2/`.
- The main dataset path is hardcoded as an absolute local Windows path in
  `hypothesis_tests/configs/config.py` / `README.txt`; it must be redefined before re-running
  anything in a new environment.
- Large binary artifacts (`.pt` checkpoints under `runs/H4.zip`, ~311 MB) were removed from this
  packaged copy to keep the archive a reasonable size — see the note above the directory tree.
- The real images/labels under `results2/H3/target_dataset/{images,labels}/test/` (~2,462 JPGs +
  2,462 YOLO `.txt` labels, ~107 MB) are now gitignored (`hypothesis_tests/.gitignore`) and were
  removed from this packaged copy too. Folder structure is preserved via `.gitkeep` files in both
  `images/test/` and `labels/test/`, and `manifest.csv` (the per-image synthetic-generation
  parameters: brightness, contrast, color gain, haze, blur, exposure, saturation, noise, rotation,
  JPEG quality, etc.) is kept intact. Re-run `scripts/H3/build_H3_target_dataset.py` to repopulate
  the two folders before actually running H3.
- `results2/MASTER_PLOTS/ALL_METRICS_INVENTORY.csv` (~296 MB, ~1.94M rows) is gitignored and was
  removed from this packaged copy — it's over GitHub's hard 100 MB per-file upload limit (which
  starts warning at 50 MB). It came from the `MASTER_PLOTS/` folder: a flattened, long-format
  inventory of per-image metrics (Precision/Recall/F1/…) pulled together from the
  `per_image_metrics.csv` files of all four hypothesis tests (`results2/H1/`, `results2/H2/`,
  `results2/H3/`, `results2/H4/` — each row's `file` column points back to the source
  `per_image_metrics.csv`). No script in this repo generates it, so it would need to be rebuilt
  from those four files if needed again. The small rollup, `METRIC_SUMMARY_ALL_HYPOTHESES.csv`
  (under 3 KB), is still tracked in the same folder.
- In `models/rtdetr_train.py` and `models/train_rtdetr_l_6gb.py`, large fully commented-out
  blocks (older draft revisions of the same script, ~1,100 lines combined) were removed from the
  top of the file; the logic and output of the live code are unchanged, just replaced with a
  short note.
