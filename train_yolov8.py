"""Explicit CLI for the supported YOLO trainer. Requires recovered dataset/environment."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Recovered 13-class data.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from unified_trainer import UnifiedTrainer
    UnifiedTrainer(args.data, model_type="yolov8", model_size="n").train(resume=args.resume)

if __name__ == "__main__":
    main()
