import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Print model layer indices for selecting CAM target layers.")
    parser.add_argument("--weights", required=True, help="Path to model weights.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    args = parser.parse_args()

    from ultralytics.nn.tasks import attempt_load_weights
    from ultralytics.utils.torch_utils import select_device

    device = select_device(args.device)
    model = attempt_load_weights(args.weights, device=device)

    for index, layer in enumerate(model.model):
        from_attr = getattr(layer, "f", "")
        layer_type = layer.__class__.__name__
        print(f"{index:>3} | from={from_attr!s:<12} | {layer_type}")


if __name__ == "__main__":
    main()
