# پایان‌نامه تشخیص پسماندهای دریایی — نسخه اصلاحات استاد

این نسخه کد، شواهد موجود و دامنه ادعاها را مطابق گزارش اصلاحات ۲۹ اکتبر ۲۰۲۶ مستند می‌کند.
تاریخ آماده‌سازی: ۱ نوامبر ۲۰۲۶. کد آموزش و استنتاج در این اصلاح دوباره اجرا نشده است.
مرجع نهایی، commit حاوی این README و PUBLICATION_MANIFEST.json است؛ SHA همان commit را از تاریخچه گیت‌هاب استناد کنید.

## دامنه علمی

| بخش | دامنه قابل پشتیبانی |
|---|---|
| H1 | بازبررسی خروجی ذخیره‌شده سه مدل؛ آموزش مستقل از صفر تأیید نشده است. |
| H2 | تفاضل جهت‌دار F1 با D_abs = clean_F1 − corrupted_F1؛ علامت abs به معنی قدرمطلق ریاضی نیست. اجرای رسمی Wilcoxon/Holm زیر بازبینی شده است. |
| H3 | فقط controlled synthetic domain shift روی همان ۲۴۶۲ تصویر آزمون؛ نام تاریخی uav_coastal_domain_shift به معنی داده واقعی مستقل پهپادی/ساحلی نیست. استنباط تاریخی H3 بر D_mix است؛ اندازه اثر Kendall W حدود ۰٫۰۰۳ و بسیار کوچک است. |
| H4 | descriptive/unverified: F1 مبدأ 0.757359 → 0.723990، اختلاف −0.033369؛ نتیجه توصیفی بازتولیدنشده است. داده زوجی خام/پیکربندی کامل در دسترس نیست؛ p و CI قدیمی تأیید نشده‌اند. بهبود در دامنه هدف آزمون‌نشده است. |

## ساختار واقعی انتشار

اسکریپت‌های آموزش در **ریشه همین مخزن** هستند؛ لازم نیست وارد پوشه models یا project_marine_debris_detection شوید.
فایل‌های H2 و CSVهای اصلی H1/H3 نیز برای بررسی مستقیم در ریشه قرار دارند.
`EVIDENCE_PATHS.json` محل متناظر هر فایل در بسته تحلیل را نشان می‌دهد.
`analysis_source_and_evidence.zip` نسخه دارای ساختار کامل اسکریپت‌های تاریخی تحلیل، داده‌های موجود و مستندات است.
برای مشاهده آن در PowerShell:

```powershell
Expand-Archive -LiteralPath .\analysis_source_and_evidence.zip -DestinationPath .\analysis_bundle
```

بسته شامل hypothesis_tests/scripts/H1 تا H4، results2، h2_output است.
بنا به انتخاب مالک، فایل Word پایان‌نامه و PowerPoint دفاع در این انتشار عمومی قرار ندارند.
این انتشار فقط کد، شواهد موجود و مستندات را شامل می‌شود؛ اصلاح اسلایدها/متن Word از همین مخزن قابل تأیید نیست.

## بازاجرای رسمی و مستقل H2 — بدون GPU و وابستگی اضافی

به Python 3.10+ و فقط کتابخانه استاندارد نیاز دارد. اجرای کنترل‌شده این نسخه با Python 3.12.13 انجام شد.
از ریشه مخزن:

```powershell
python h2_wilcoxon_holm_verified_v2.py per_image_degradation.csv --out h2_recheck
```

خروجی شش‌ردیفی `h2_recheck/h2_inferential_results.csv` باید با فایل ریشه `h2_inferential_results.csv` مقایسه شود.
قرارداد: میانگین شش شرط برای هر تصویر، n=2462؛ اختلاف‌های زوجی با گردکردن ۱۲ رقم؛ حذف صفر از رتبه‌ها؛
Wilcoxon دوطرفه با تقریب نرمال و تصحیح tie، بدون تصحیح پیوستگی؛ Holm جداگانه در هر خانواده سه‌مقایسه‌ای.
صفرهای تحلیل اصلی به ترتیب RT-DETR-L/RTMDet-Tiny، RT-DETR-L/YOLOv8n، RTMDet-Tiny/YOLOv8n: **267، 351، 260**.
سه ردیف pooled با n=14772 صرفاً تحلیل حساسیت‌اند؛ شش شرط هر تصویر مستقل نیستند.
نسخه رسمی یگانه `h2_wilcoxon_holm_verified_v2.py` است؛ ورودی `degradation` ترکیبی را به جای D_abs قبول نمی‌کند.
رابط قدیمی H2 داخل بسته فقط به همین پیاده‌سازی ارجاع می‌دهد.

SHA-256 ورودی: `15d46b585c72913e53275e1314552359f664ef2c96a979356b572eadc3ad278b`

SHA-256 اسکریپت رسمی: `f9461dbcfcee9b82d4f84ff872514e4433c491b77edd0e2ef17f32d24a7e76c9`

SHA-256 خروجی شش‌ردیفی: `72277102ee428104ec35af29ce8a9cadf1a50e84cd5731f2fa10f3f031c1eda6`

`run_manifest.json` اجرای ثبت‌شده قبلی را نگه می‌دارد؛ هر بازاجرا manifest زمانی جدید خود را می‌سازد.
`VALIDATION.json` نتیجه کنترل انتشار را ثبت می‌کند. یکسانی خروجی H2 در این آماده‌سازی مجدداً کنترل شد.

## مسیرهای واقعی آموزش و فایل‌های Legacy

| مدل | مسیر مرجع مستندشده در پایان‌نامه | وضعیت |
|---|---|---|
| YOLOv8n | unified_trainer.py؛ CLI جدید train_yolov8.py | فقط YOLO؛ درخواست غیر YOLO پیش از ساخت خروجی با NotImplementedError متوقف می‌شود. |
| RT-DETR-L | train_rtdetr_l_6gb.py | اسکریپت اختصاصی؛ وزن ثبت‌شده ارزیابی last.pt است، نه best اثبات‌شده. |
| RTMDet-Tiny | train_rtmdet_tiny.py | اسکریپت اختصاصی MMDetection. |

`rtmdet_train.py` یک wrapper قدیمی است؛ تولید تاریخچه/سنجه نمایشی حذف و ورودی‌های اجرای آن غیرفعال شده‌اند.
`rtdetr_train.py` و `run_rtdetr.py` پیش‌نویس‌اند و مسیر نتایج پایان‌نامه محسوب نمی‌شوند.
`yolov8_train.py` فقط نام سازگاری برای UnifiedTrainer است؛ پیاده‌سازی آماری یا آموزشی موازی ندارد.
`__init__.py` دیگر کلاس RTDETRTrainer ناموجود را import نمی‌کند.
فایل‌های pyc قدیمی از قبل در تاریخچه هستند و مرجع این اصلاح نیستند؛ .gitignore از افزودن نمونه‌های جدید جلوگیری می‌کند.

محیط اصلی کامل آموزش و تمام نسخه‌های کتابخانه‌ها بازیابی نشده‌اند. `config.py` تنظیم قابل‌حمل جدید با پیش‌فرض‌های صریح کد است و بازسازی قطعی تنظیمات تاریخی محسوب نمی‌شود.
وابستگی‌های مدل‌ها شامل PyTorch/Ultralytics و برای RTMDet شامل MMDetection/MMEngine هستند؛ سازگاری محیط باید جداگانه بررسی شود.
فرمان‌های زیر **فقط پس از بازیابی دیتاست و محیط مدل‌ها** هستند؛ در این اصلاح اجرا نشده‌اند:

```powershell
$env:MARINE_DATASET_ROOT = 'C:\path\to\recovered_dataset'
python train_yolov8.py --data "$env:MARINE_DATASET_ROOT/data.yaml"
python train_rtdetr_l_6gb.py
python train_rtmdet_tiny.py
```

برای ادامه آموزش RT-DETR/RTMDet از --resume استفاده کنید. --test در RT-DETR آزمون کوتاه آموزش و در RTMDet ارزیابی TEST است؛ این دو معنا یکسان نیستند.
گزینه‌های MARINE_PROJECT_ROOT و MARINE_RESULTS_ROOT نیز به‌ترتیب مسیر ریشه اسکریپت‌های اختصاصی و خروجی YOLO را قابل تنظیم می‌کنند.

## فرمان‌های تاریخی H1 تا H4 در بسته تحلیل

پس از استخراج آرشیو و `cd analysis_bundle`، مقادیر ALL_CAPS را با فایل‌های واقعی بازیابی‌شده جایگزین کنید.
فرمان‌های H1–H3 به وزن‌ها/تصاویر/برچسب‌ها و محیط مناسب نیاز دارند و بازاجرای آن‌ها در این اصلاح انجام نشده است:

```powershell
python hypothesis_tests/scripts/H1/H1_controlled_baseline_map_cached.py --test-images SOURCE_IMAGES --test-labels SOURCE_LABELS --yolov8 YOLO_BEST_PT --rtdetr RTDETR_LAST_PT --rtmdet RTMDET_EPOCH146_PTH --rtmdet-config RTMDET_CONFIG --output H1_NEW_OUTPUT --imgsz 640 --conf 0.25 --iou 0.50
python hypothesis_tests/scripts/H2/run_h2_complete.py --test-images SOURCE_IMAGES --test-labels SOURCE_LABELS --corruption-root CORRUPTION_IMAGES --yolov8 YOLO_BEST_PT --rtdetr RTDETR_LAST_PT --rtmdet RTMDET_EPOCH146_PTH --rtmdet-config RTMDET_CONFIG --imgsz 640 --conf 0.25 --iou 0.50
python hypothesis_tests/scripts/H3/run_h3_complete.py --source-images SOURCE_IMAGES --source-labels SOURCE_LABELS --target-images SYNTHETIC_IMAGES --target-labels SYNTHETIC_LABELS --yolov8 YOLO_BEST_PT --rtdetr RTDETR_LAST_PT --rtmdet RTMDET_EPOCH146_PTH --rtmdet-config RTMDET_CONFIG --pairing-mode paired --imgsz 640 --conf 0.25 --iou 0.50
```

H4 رابط CLI ندارد. پس از بازیابی ورودی‌ها باید PROJECT_ROOT، DATASET_ROOT و BASE_MODEL در بخش CONFIGURATION تنظیم شوند:

```powershell
python hypothesis_tests/scripts/H4/run_h4_end_to_end.py
```

پارامترهای تاریخی H4: imgsz=640، conf=0.25، IoU=0.50، seed=42، سقف 20 دوره، batch=10، lr0=1e-4؛ لاگ موجود 14 دوره دارد.
این فرمان جایگزین آزمون مستقل بهبود هدف نیست. ورودی قدیمی run_h4_statistics.py برای این نسخه غیرفعال است.
سایر اسکریپت‌های تکمیلی آرشیو مسیرها/نام‌های تاریخی دارند و مرجع ادعاهای تازه یا بازاجرای تأییدشده نیستند.

## شواهد موجود و غایب

موجود: سورس‌های آموزش؛ سورس تاریخی ارزیابی؛ CSVهای H1–H3؛ manifest تبدیل مصنوعی H3؛
کد و خروجی رسمی H2؛ لاگ 14 دوره H4؛ هش فایل‌های انتشار.

غایب: وزن نهایی مدل‌های H1–H4؛ تصاویر/برچسب کامل اصلی و هدف H3؛ داده خام زوجی H4؛
خروجی کامل محیط آموزش؛ لاگ‌های کامل انتخاب checkpoint. لینک قانونی پایدار این موارد نیز ارائه نشده است؛ لینک یا هش ساختگی ساخته نشده است.

محدودیت‌ها: ۱۳ کلاس آموزشی در برابر ۱۲ شناسه AP؛ علت غیبت کلاس ۹ بدون برچسب خام نامعلوم است.
سنجه mAP سفارشی با ۱۰۱ نقطه و conf≥0.25، تفاوت ظرفیت و recipe مدل‌ها، تک‌بذری‌بودن آموزش،
وابستگی احتمالی فریم‌های هم‌منبع، نویز H2 بدون بذر صریح و نبود benchmark مشترک FPS باید در تفسیر حفظ شوند.
وجود CSV امکان بازتحلیل خروجی را می‌دهد و به معنی تأیید مستقل آموزش از صفر نیست.

## English scope

H3 is a controlled synthetic transformation of the same 2462 test images, never an independent real UAV/coastal dataset.
H4 is descriptive/unverified: source F1 0.757359 to 0.723990, delta -0.033369; target improvement is untested.
No verified H4 p-value/CI or best-checkpoint claim is made. The canonical H2 script is standard-library-only;
all six output rows were rechecked byte-for-byte. GPU training/inference was not rerun.
See PUBLICATION_MANIFEST.json for SHA-256 inventory and VALIDATION.json for checks.
