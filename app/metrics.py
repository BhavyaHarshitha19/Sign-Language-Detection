"""
Internal performance metrics logger (NOT exposed in UI).
Logs per-sentence metrics to logs/metrics.jsonl and maintains session averages.

Metrics computed:
  - gesture_accuracy  : % correct gestures vs ground-truth labels (if provided)
  - bleu_score        : BLEU-1 similarity between generated and reference sentence
  - wer               : Word Error Rate (insertions + deletions + substitutions)
  - latency_ms        : Time from first token to sentence generation
  - translation_score : Simple token-overlap proxy for semantic correctness
"""

import os
import json
import math
import time
import logging
from pathlib import Path
from collections import Counter
from typing import Optional

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "metrics.jsonl"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("isl.metrics")
logging.basicConfig(level=logging.INFO)

# ── Session running averages ──────────────────────────────────────────────
_session: dict = {
    "count":              0,
    "bleu_sum":           0.0,
    "wer_sum":            0.0,
    "latency_sum":        0.0,
    "gesture_acc_sum":    0.0,
    "gesture_acc_count":  0,
}


# ── BLEU-1 (unigram) ──────────────────────────────────────────────────────
def _bleu1(hypothesis: str, reference: str) -> float:
    hyp  = hypothesis.lower().split()
    ref  = reference.lower().split()
    if not hyp:
        return 0.0
    ref_counts = Counter(ref)
    clip = sum(min(cnt, ref_counts[w]) for w, cnt in Counter(hyp).items())
    precision = clip / len(hyp)
    # Brevity penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    return round(bp * precision, 4)


# ── Word Error Rate ───────────────────────────────────────────────────────
def _wer(hypothesis: str, reference: str) -> float:
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not ref:
        return 0.0
    # Dynamic programming edit distance
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return round(d[len(ref)][len(hyp)] / len(ref), 4)


# ── Gesture accuracy ──────────────────────────────────────────────────────
def _gesture_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    if not ground_truth:
        return 0.0
    correct = sum(p == g for p, g in zip(predicted, ground_truth))
    return round(correct / len(ground_truth), 4)


# ── Public API ────────────────────────────────────────────────────────────
def log_sentence(
    tokens:           list[str],
    sentence:         str,
    latency_ms:       float,
    reference:        Optional[str]       = None,
    ground_truth_tokens: Optional[list[str]] = None,
) -> dict:
    """
    Compute and persist metrics for one completed sentence.
    Returns the metrics dict (for internal use / debugging only).
    """
    metrics: dict = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tokens":      tokens,
        "sentence":    sentence,
        "latency_ms":  round(latency_ms, 1),
        "bleu":        None,
        "wer":         None,
        "gesture_accuracy": None,
        "translation_score": None,
    }

    if reference:
        metrics["bleu"] = _bleu1(sentence, reference)
        metrics["wer"]  = _wer(sentence, reference)
        # Simple translation score: token overlap ratio
        gen_words = set(sentence.lower().split())
        ref_words = set(reference.lower().split())
        overlap   = len(gen_words & ref_words)
        metrics["translation_score"] = round(overlap / max(len(ref_words), 1), 4)

    if ground_truth_tokens:
        metrics["gesture_accuracy"] = _gesture_accuracy(tokens, ground_truth_tokens)

    # Update session averages
    _session["count"]       += 1
    _session["latency_sum"] += latency_ms
    if metrics["bleu"] is not None:
        _session["bleu_sum"] += metrics["bleu"]
        _session["wer_sum"]  += metrics["wer"]
    if metrics["gesture_accuracy"] is not None:
        _session["gesture_acc_sum"]   += metrics["gesture_accuracy"]
        _session["gesture_acc_count"] += 1

    n = _session["count"]
    metrics["session_avg"] = {
        "avg_latency_ms": round(_session["latency_sum"] / n, 1),
        "avg_bleu":       round(_session["bleu_sum"] / n, 4) if n else 0,
        "avg_wer":        round(_session["wer_sum"]  / n, 4) if n else 0,
        "avg_gesture_acc": round(
            _session["gesture_acc_sum"] / _session["gesture_acc_count"], 4
        ) if _session["gesture_acc_count"] else None,
        "total_sentences": n,
    }

    # Persist to JSONL
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    logger.info("[METRICS] latency=%.0fms bleu=%s wer=%s",
                latency_ms, metrics["bleu"], metrics["wer"])
    return metrics
