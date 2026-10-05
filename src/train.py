"""Train, tune on validation, evaluate once on test, save the model and its metrics."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from src.pipeline import OUT_OF_SCOPE, build_pipeline, load, predict_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/clinc150.json")
    parser.add_argument("--out", default="model")
    args = parser.parse_args()

    X_train, y_train = load(args.data, "train")
    X_val, y_val = load(args.data, "val")

    # 1) choose C on the validation split (the test split stays untouched)
    best_c, best_acc = None, -1.0
    for c in (10.0, 30.0):
        acc = accuracy_score(y_val, build_pipeline(c).fit(X_train, y_train).predict(X_val))
        print(f"C={c}: val accuracy {acc:.3f}")
        if acc > best_acc:
            best_c, best_acc = c, acc
    model = build_pipeline(best_c).fit(X_train, y_train)

    # 2) choose the out-of-scope threshold on validation: keep in-scope accuracy as high as
    #    possible while still catching at least 60% of out-of-scope requests
    oos_val, _ = load(args.data, "oos_val")
    candidates = []
    for t in np.arange(0.05, 0.6, 0.025):
        in_acc = accuracy_score(y_val, predict_with_fallback(model, X_val, t))
        oos_rec = np.mean([p == OUT_OF_SCOPE for p in predict_with_fallback(model, oos_val, t)])
        if oos_rec >= 0.6:
            candidates.append((in_acc, t))
    threshold = max(candidates)[1]

    # 3) evaluate ONCE on the test splits
    X_test, y_test = load(args.data, "test")
    oos_test, _ = load(args.data, "oos_test")
    metrics = {
        "C": best_c,
        "threshold": round(float(threshold), 3),
        "test_accuracy": round(accuracy_score(y_test, predict_with_fallback(model, X_test, threshold)), 4),
        "oos_recall": round(float(np.mean([p == OUT_OF_SCOPE for p in predict_with_fallback(model, oos_test, threshold)])), 4),
    }
    print(json.dumps(metrics, indent=2))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()