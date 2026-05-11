"""
ISL Model Evaluation Script
=============================
Loads the trained model and a labeled test CSV, then computes and saves:
  - Accuracy, Precision, Recall, F1 (per class + macro/weighted)
  - Confusion matrix  → logs/eval_confusion_matrix.csv
  - Full report       → logs/eval_report.txt
  - BLEU / WER        → logs/eval_translation.csv  (if sentence pairs provided)

Usage:
    # Gesture-level evaluation (uses same dataset by default)
    python scripts/evaluate_model.py

    # With a separate held-out test CSV
    python scripts/evaluate_model.py --test-csv data/isl_test.csv

    # With sentence-level ground truth for BLEU/WER
    python scripts/evaluate_model.py --sentences data/sentences.csv
    # sentences.csv format: two columns — tokens (space-separated), reference
"""

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = Path("models")
LOG_DIR   = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ── Metric helpers ────────────────────────────────────────────────────────
def bleu1(hypothesis: str, reference: str) -> float:
    hyp, ref = hypothesis.lower().split(), reference.lower().split()
    if not hyp:
        return 0.0
    ref_c  = Counter(ref)
    clip   = sum(min(c, ref_c[w]) for w, c in Counter(hyp).items())
    prec   = clip / len(hyp)
    bp     = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    return round(bp * prec, 4)


def wer(hypothesis: str, reference: str) -> float:
    hyp, ref = hypothesis.lower().split(), reference.lower().split()
    if not ref:
        return 0.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return round(d[len(ref)][len(hyp)] / len(ref), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv",   default="data/isl_dataset.csv")
    parser.add_argument("--sentences",  default=None,
                        help="CSV with columns: tokens, reference")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    clf_path = MODEL_DIR / "isl_classifier.pkl"
    le_path  = MODEL_DIR / "label_encoder.pkl"
    if not clf_path.exists():
        raise FileNotFoundError("Model not found. Run scripts/train_model.py first.")

    with open(clf_path, "rb") as f: clf = pickle.load(f)
    with open(le_path,  "rb") as f: le  = pickle.load(f)

    # ── Gesture-level evaluation ──────────────────────────────────────────
    df = pd.read_csv(args.test_csv)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y_true_str = df["label"].values
    y_true     = le.transform(y_true_str)
    y_pred     = clf.predict(X)

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=le.classes_, digits=4
    )
    cm     = confusion_matrix(y_true, y_pred)
    cm_df  = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print(f"\n[GESTURE EVALUATION]")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n{report}")

    # Save
    with open(LOG_DIR / "eval_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")
    cm_df.to_csv(LOG_DIR / "eval_confusion_matrix.csv")
    print(f"  Saved → logs/eval_report.txt, logs/eval_confusion_matrix.csv")

    # ── Sentence-level evaluation (optional) ─────────────────────────────
    if args.sentences and Path(args.sentences).exists():
        rows = []
        with open(args.sentences, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hyp = row.get("generated", "").strip()
                ref = row.get("reference", "").strip()
                if hyp and ref:
                    rows.append({
                        "generated": hyp,
                        "reference": ref,
                        "bleu":      bleu1(hyp, ref),
                        "wer":       wer(hyp, ref),
                    })

        if rows:
            trans_df = pd.DataFrame(rows)
            avg_bleu = trans_df["bleu"].mean()
            avg_wer  = trans_df["wer"].mean()
            print(f"\n[TRANSLATION EVALUATION]")
            print(f"  Avg BLEU : {avg_bleu:.4f}")
            print(f"  Avg WER  : {avg_wer:.4f}")
            trans_df.to_csv(LOG_DIR / "eval_translation.csv", index=False)
            print(f"  Saved → logs/eval_translation.csv")


if __name__ == "__main__":
    main()
