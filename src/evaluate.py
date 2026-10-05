"""Quality gate: exit with an error (which fails CI) if the model is not good enough."""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="model/metrics.json")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-oos-recall", type=float, default=0.50)
    args = parser.parse_args()

    m = json.loads(Path(args.metrics).read_text())
    failures = []
    if m["test_accuracy"] < args.min_accuracy:
        failures.append(f"test accuracy {m['test_accuracy']:.3f} < {args.min_accuracy}")
    if m["oos_recall"] < args.min_oos_recall:
        failures.append(f"out-of-scope recall {m['oos_recall']:.3f} < {args.min_oos_recall}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        sys.exit(1)
    print(f"PASS: accuracy {m['test_accuracy']:.3f}, out-of-scope recall {m['oos_recall']:.3f}")


if __name__ == "__main__":
    main()