import argparse
import csv
import os
import statistics
import time
from pathlib import Path


def weight_size_mb(path):
    return os.stat(path).st_size / 1024 / 1024


def load_model(weights, device, fuse=True):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import attempt_load_weights

    if weights.endswith(".pt"):
        model = attempt_load_weights(weights, device=device, fuse=fuse)
    else:
        model = YOLO(weights).model.to(device)
        if fuse and hasattr(model, "fuse"):
            model.fuse()
    model.eval()
    return model


def measure_latency(weights, device, imgsz, batch, warmup, iterations, half):
    import torch

    model = load_model(weights, device)
    inputs = torch.randn((batch, 3, imgsz, imgsz), device=device)

    if half:
        model = model.half()
        inputs = inputs.half()

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)

        timings = []
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

    latency_s = sum(timings) / (iterations * batch)
    std_s = statistics.pstdev(timings) / batch if len(timings) > 1 else 0.0
    return latency_s * 1000, std_s * 1000, 1.0 / latency_s


def validate(weights, data, split, imgsz, batch, device, project, name):
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=False,
    )

    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def parse_model_spec(spec):
    parts = spec.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("model spec must be NAME=WEIGHTS")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate FPS/latency and optional validation CSVs for PMF-YOLO revision."
    )
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True, help="NAME=path/to/best.pt")
    parser.add_argument("--device", default="0", help="CUDA device id, e.g. 0, or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--half", action="store_true", help="Use FP16 during latency measurement.")
    parser.add_argument("--data", default="", help="Optional dataset yaml for validation.")
    parser.add_argument("--split", default="test", help="Validation split: val or test.")
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--out", default="runs/revision/benchmark.csv")
    parser.add_argument("--project", default="runs/revision/val")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.torch_utils import select_device

    device = select_device(args.device, batch=args.batch)

    rows = []
    for name, weights in args.model:
        latency_ms, latency_std_ms, fps = measure_latency(
            weights=weights,
            device=device,
            imgsz=args.imgsz,
            batch=args.batch,
            warmup=args.warmup,
            iterations=args.iterations,
            half=args.half,
        )
        row = {
            "model": name,
            "weights": weights,
            "weight_size_mb": f"{weight_size_mb(weights):.2f}" if weights.endswith(".pt") else "",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "device": str(device),
            "half": args.half,
            "latency_ms": f"{latency_ms:.4f}",
            "latency_std_ms": f"{latency_std_ms:.4f}",
            "fps": f"{fps:.2f}",
            "precision": "",
            "recall": "",
            "map50": "",
            "map50_95": "",
            "map50_per_param": "",
            "map50_95_per_param": "",
        }

        if args.data:
            vals = validate(
                weights=weights,
                data=args.data,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.val_batch,
                device=args.device,
                project=args.project,
                name=f"{name}_{args.split}",
            )
            row.update({key: f"{value:.4f}" for key, value in vals.items()})
            if row["weight_size_mb"]:
                params_proxy = float(row["weight_size_mb"])
                row["map50_per_param"] = f"{vals['map50'] / params_proxy:.4f}"
                row["map50_95_per_param"] = f"{vals['map50_95'] / params_proxy:.4f}"

        rows.append(row)
        print(f"{name}: latency={row['latency_ms']} ms, FPS={row['fps']}, mAP50={row['map50']}")

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
