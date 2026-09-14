import os
import sys
import platform
import gc
import traceback

print("=" * 80)
print("RT-DETR DIAGNOSTIC TEST")
print("=" * 80)

# ============================================================
# 1. SYSTEM
# ============================================================

print("\n[1] SYSTEM")
print("-" * 80)

print("OS     :", platform.platform())
print("Python :", sys.version.replace("\n", " "))

# ============================================================
# 2. PYTORCH
# ============================================================

try:
    import torch

    print("\n[2] PYTORCH")
    print("-" * 80)

    print("PyTorch        :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    print("CUDA version   :", torch.version.cuda)

    if torch.cuda.is_available():

        print("GPU count      :", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):

            props = torch.cuda.get_device_properties(i)

            print(f"\nGPU {i}")
            print("Name           :", props.name)
            print(
                "VRAM           :",
                round(props.total_memory / 1024**3, 2),
                "GB"
            )
            print(
                "Compute Cap.   :",
                f"{props.major}.{props.minor}"
            )

            try:
                free, total = torch.cuda.mem_get_info(i)

                print(
                    "Free VRAM      :",
                    round(free / 1024**3, 2),
                    "GB"
                )

                print(
                    "Total VRAM     :",
                    round(total / 1024**3, 2),
                    "GB"
                )

            except Exception as e:
                print("VRAM info error:", e)

except Exception as e:

    print("PyTorch ERROR:")
    print(e)

    sys.exit(1)


# ============================================================
# 3. ULTRALYTICS
# ============================================================

print("\n[3] ULTRALYTICS")
print("-" * 80)

try:

    import ultralytics

    print(
        "Ultralytics version :",
        ultralytics.__version__
    )

except Exception as e:

    print("Ultralytics ERROR:")
    print(e)

    sys.exit(1)


# ============================================================
# 4. MODEL FILE CHECK
# ============================================================

print("\n[4] MODEL FILE CHECK")
print("-" * 80)

models = [
    "rtdetr-r18.pt",
    "rtdetr-l.pt",
    "rtdetr-x.pt",
]

for name in models:

    if os.path.exists(name):

        size = os.path.getsize(name) / 1024**2

        print(
            f"[FOUND]   {name:<20} "
            f"{size:.2f} MB"
        )

    else:

        print(
            f"[MISSING] {name}"
        )


# ============================================================
# 5. CURRENT TRAINER LOGIC
# ============================================================

print("\n[5] CURRENT TRAINER LOGIC")
print("-" * 80)


def current_trainer_selection(model_type, model_size):

    if model_type == "yolov8":

        return f"yolov{model_size}.pt"

    elif model_type == "rtdetr":

        if model_size in ["l", "x"]:

            return f"rtdetr-{model_size}.pt"

        else:

            return "rtdetr-l.pt"

    elif model_type == "rtmdet":

        if model_size in ["l", "x"]:

            return f"rtmdet-{model_size}.pt"

        else:

            return "rtmdet-l.pt"

    return None


for size in ["r18", "l", "x", "n", "s"]:

    selected = current_trainer_selection(
        "rtdetr",
        size
    )

    print(
        f"Requested: {size:<5} "
        f"--> Selected: {selected}"
    )


# ============================================================
# 6. LOAD MODELS
# ============================================================

from ultralytics import YOLO


def inspect_model(model_name):

    print("\n" + "=" * 80)
    print("LOADING:", model_name)
    print("=" * 80)

    try:

        model = YOLO(model_name)

        print("\nLOAD SUCCESS")

        print(
            "Model class :",
            type(model.model)
        )

        print(
            "Task        :",
            model.task
        )

        # --------------------------------------------
        # Parameters
        # --------------------------------------------

        try:

            total_params = sum(
                p.numel()
                for p in model.model.parameters()
            )

            trainable_params = sum(
                p.numel()
                for p in model.model.parameters()
                if p.requires_grad
            )

            print("\nParameters")

            print(
                "Total       :",
                f"{total_params:,}"
            )

            print(
                "Trainable   :",
                f"{trainable_params:,}"
            )

            print(
                "Millions    :",
                round(total_params / 1e6, 2),
                "M"
            )

        except Exception as e:

            print(
                "Parameter count error:",
                e
            )

        # --------------------------------------------
        # YAML
        # --------------------------------------------

        try:

            yaml_data = model.model.yaml

            print("\nModel YAML information")

            if isinstance(yaml_data, dict):

                for key in yaml_data:

                    if key not in [
                        "backbone",
                        "head"
                    ]:

                        print(
                            f"{key}:",
                            yaml_data[key]
                        )

            else:

                print(yaml_data)

        except Exception as e:

            print(
                "YAML error:",
                e
            )

        return model

    except Exception as e:

        print("\nLOAD FAILED")

        print(
            "Error:",
            repr(e)
        )

        traceback.print_exc()

        return None


# ============================================================
# 7. R18
# ============================================================

r18 = inspect_model(
    "rtdetr-r18.pt"
)


# ============================================================
# 8. L
# ============================================================

rtdetr_l = inspect_model(
    "rtdetr-l.pt"
)


# ============================================================
# 9. X
# ============================================================

rtdetr_x = inspect_model(
    "rtdetr-x.pt"
)


# ============================================================
# 10. PARAMETER COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("[6] PARAMETER COMPARISON")
print("=" * 80)


def count_parameters(model):

    if model is None:
        return None

    try:

        return sum(
            p.numel()
            for p in model.model.parameters()
        )

    except Exception:

        return None


comparison = [
    ("RT-DETR-R18", r18),
    ("RT-DETR-L", rtdetr_l),
    ("RT-DETR-X", rtdetr_x),
]

for name, model in comparison:

    params = count_parameters(model)

    if params is None:

        print(
            f"{name:<15}: NOT AVAILABLE"
        )

    else:

        print(
            f"{name:<15}: "
            f"{params:,} "
            f"({params / 1e6:.2f} M)"
        )


# ============================================================
# 11. FORWARD TEST
# ============================================================

print("\n" + "=" * 80)
print("[7] GPU FORWARD TEST")
print("=" * 80)

print("""
Test configuration:

Batch     = 1
Image size = 512 x 512
Training  = NO
Dataset   = NO
""")

if torch.cuda.is_available():

    device = "cuda:0"

    for name, model in comparison:

        if model is None:

            print(
                f"\n{name}: SKIPPED"
            )

            continue

        print("\n" + "-" * 80)
        print("Testing:", name)
        print("-" * 80)

        try:

            gc.collect()
            torch.cuda.empty_cache()

            model.model.to(device)
            model.model.eval()

            x = torch.zeros(
                (1, 3, 512, 512),
                dtype=torch.float32,
                device=device
            )

            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():

                output = model.model(x)

            torch.cuda.synchronize()

            current_memory = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved_memory = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            peak_memory = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                "[SUCCESS] Forward pass"
            )

            print(
                "Current VRAM :",
                round(current_memory, 2),
                "GB"
            )

            print(
                "Reserved VRAM:",
                round(reserved_memory, 2),
                "GB"
            )

            print(
                "Peak VRAM    :",
                round(peak_memory, 2),
                "GB"
            )

            del x
            del output

            model.model.cpu()

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:

            print(
                "[FAILED]"
            )

            print(
                "Error:",
                repr(e)
            )

            traceback.print_exc()

            try:
                model.model.cpu()
            except:
                pass

            gc.collect()
            torch.cuda.empty_cache()

else:

    print(
        "CUDA is not available."
    )


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

if r18 is None:

    print("""
❌ RT-DETR-R18 در محیط فعلی Load نشد.

بنابراین Trainer فعلی نمی‌تواند به شکل استاندارد
از rtdetr-r18.pt استفاده کند.
""")

else:

    print("""
✅ RT-DETR-R18 با موفقیت Load شد.
""")


print("""
مهم‌ترین خط بالا:

Requested: r18 --> Selected: ...

اگر نتیجه این باشد:

Requested: r18 --> Selected: rtdetr-l.pt

پس Trainer فعلی تو برای R18،
در واقع RT-DETR-L را اجرا می‌کند.
""")

print("=" * 80)
print("TEST FINISHED")
print("=" * 80)