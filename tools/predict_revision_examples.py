import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate detection examples for PMF-YOLO revision figures.")
    parser.add_argument("--weights", required=True, help="Path to model weights.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--out", default="runs/revision/predict", help="Output project directory.")
    parser.add_argument("--name", default="examples", help="Run name under output directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=args.out,
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        line_width=args.line_width,
        show_conf=not args.hide_conf,
        show_labels=not args.hide_labels,
    )
    print(f"Saved predictions to: {Path(args.out) / args.name}")


if __name__ == "__main__":
    main()
