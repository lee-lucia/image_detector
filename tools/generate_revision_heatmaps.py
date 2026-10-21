import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_layers(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Generate CAM heatmaps for PMF-YOLO revision figures.")
    parser.add_argument("--weights", required=True, help="Path to model weights, e.g. runs/train/pmf/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--out", default="runs/revision/heatmaps", help="Output directory.")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, or cpu")
    parser.add_argument(
        "--method",
        default="LayerCAM",
        choices=["GradCAMPlusPlus", "GradCAM", "XGradCAM", "EigenCAM", "HiResCAM", "LayerCAM", "RandomCAM", "EigenGradCAM"],
    )
    parser.add_argument("--layers", default="30,35,40,45", help="Comma-separated model layer indices.")
    parser.add_argument("--backward-type", default="class", choices=["class", "box", "all"])
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--show-box", action="store_true")
    parser.add_argument("--renormalize", action="store_true")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete output directory before writing.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images under source directory.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source}. "
            "Create the directory and put selected images in it, or pass a valid image path."
        )
    if source.is_dir():
        pattern = "**/*" if args.recursive else "*"
        image_files = sorted(p for p in source.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        image_files = [source] if source.suffix.lower() in IMAGE_EXTS else []

    if not image_files:
        raise FileNotFoundError(f"No image files found in source path: {source}")

    out_dir = Path(args.out)
    if out_dir.exists() and not args.keep_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from heatmap import yolov8_heatmap

    model = yolov8_heatmap(
        weight=args.weights,
        device=args.device,
        method=args.method,
        layer=parse_layers(args.layers),
        backward_type=args.backward_type,
        conf_threshold=args.conf,
        ratio=args.ratio,
        show_box=args.show_box,
        renormalize=args.renormalize,
    )
    for image_path in image_files:
        if source.is_dir() and args.recursive:
            save_path = out_dir / image_path.relative_to(source)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        elif source.is_dir():
            save_path = out_dir / image_path.name
        else:
            save_path = out_dir / "result.png"
        model.process(str(image_path), str(save_path))
    print(f"Saved heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()
