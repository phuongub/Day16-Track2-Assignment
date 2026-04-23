"""
benchmark.py — LightGBM Benchmark on Credit Card Fraud Detection Dataset
Target instance: r5.2xlarge (8 vCPU, 32 GB RAM)

Metrics collected (Section 7.6):
  - Thời gian load data         (Data loading time)
  - Thời gian training           (Training time)
  - Best iteration
  - AUC-ROC
  - Accuracy
  - F1-Score
  - Precision
  - Recall
  - Inference latency (1 row)
  - Inference throughput (1000 rows)

Output: benchmark_result.json
"""

import time
import json
import platform
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_result.json")

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 577,   # ~ratio of negatives to positives in fraud dataset
    "n_jobs": -1,              # use all vCPUs on r5.2xlarge (8 cores)
    "random_state": 42,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 50
TEST_SIZE = 0.2
INFERENCE_WARMUP_ROWS = 1
INFERENCE_THROUGHPUT_ROWS = 1000
INFERENCE_LATENCY_REPS = 100   # repeat single-row inference N times, take median


def separator(char="─", width=60):
    print(char * width)


def banner(text):
    separator("═")
    print(f"  {text}")
    separator("═")


# ──────────────────────────────────────────────
# 1. ENVIRONMENT INFO
# ──────────────────────────────────────────────
def print_env_info():
    banner("ENVIRONMENT INFO")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  LightGBM : {lgb.__version__}")
    print(f"  NumPy    : {np.__version__}")
    print(f"  Pandas   : {pd.__version__}")
    print(f"  Platform : {platform.platform()}")
    cpu_count = os.cpu_count()
    print(f"  CPUs     : {cpu_count}")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"  RAM      : {ram_gb:.1f} GB")
    except ImportError:
        print("  RAM      : (install psutil for RAM info)")
    print()


# ──────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────
def load_data():
    banner("STEP 1 — LOAD DATA")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        print("  Please download it first:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/")
        sys.exit(1)

    print(f"  Loading: {DATA_PATH}")
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    load_time = time.perf_counter() - t0

    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {len(df.columns)}")
    fraud_pct = df["Class"].mean() * 100
    print(f"  Fraud rows  : {df['Class'].sum():,}  ({fraud_pct:.4f}%)")
    print(f"  ✓ Load time : {load_time:.3f}s")
    print()
    return df, load_time


# ──────────────────────────────────────────────
# 3. PREPROCESS & SPLIT
# ──────────────────────────────────────────────
def prepare_data(df):
    banner("STEP 2 — PREPROCESS & SPLIT")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"  Train samples : {len(X_train):,}  (fraud: {y_train.sum():,})")
    print(f"  Test  samples : {len(X_test):,}   (fraud: {y_test.sum():,})")
    print()
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 4. TRAIN
# ──────────────────────────────────────────────
def train_model(X_train, X_test, y_train, y_test):
    banner("STEP 3 — TRAIN LightGBM")
    print("  Params:")
    for k, v in LGBM_PARAMS.items():
        print(f"    {k}: {v}")
    print()

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - t0

    best_iter = model.best_iteration_
    print(f"\n  ✓ Training time  : {train_time:.2f}s")
    print(f"  ✓ Best iteration : {best_iter}")
    print()
    return model, train_time, best_iter


# ──────────────────────────────────────────────
# 5. EVALUATE
# ──────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    banner("STEP 4 — EVALUATE")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc    = roc_auc_score(y_test, y_proba)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred)
    rec    = recall_score(y_test, y_pred)

    print(f"  AUC-ROC   : {auc:.6f}")
    print(f"  Accuracy  : {acc:.6f}")
    print(f"  F1-Score  : {f1:.6f}")
    print(f"  Precision : {prec:.6f}")
    print(f"  Recall    : {rec:.6f}")
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print()
    return auc, acc, f1, prec, rec


# ──────────────────────────────────────────────
# 6. INFERENCE BENCHMARK
# ──────────────────────────────────────────────
def benchmark_inference(model, X_test):
    banner("STEP 5 — INFERENCE BENCHMARK")

    # ── Single-row latency ──────────────────────
    single_row = X_test.iloc[[0]]

    # Warm-up (avoid JIT / cache cold effects)
    for _ in range(5):
        model.predict_proba(single_row)

    latencies = []
    for _ in range(INFERENCE_LATENCY_REPS):
        t0 = time.perf_counter()
        model.predict_proba(single_row)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latency_median_ms = float(np.median(latencies))
    latency_p95_ms    = float(np.percentile(latencies, 95))
    latency_p99_ms    = float(np.percentile(latencies, 99))

    print(f"  Single-row inference ({INFERENCE_LATENCY_REPS} reps):")
    print(f"    Median  : {latency_median_ms:.3f} ms")
    print(f"    P95     : {latency_p95_ms:.3f} ms")
    print(f"    P99     : {latency_p99_ms:.3f} ms")
    print()

    # ── Throughput (1000 rows) ──────────────────
    batch = X_test.iloc[:INFERENCE_THROUGHPUT_ROWS]

    # Warm-up
    model.predict_proba(batch)

    t0 = time.perf_counter()
    model.predict_proba(batch)
    throughput_time_ms = (time.perf_counter() - t0) * 1000
    rows_per_sec = (INFERENCE_THROUGHPUT_ROWS / throughput_time_ms) * 1000

    print(f"  Batch ({INFERENCE_THROUGHPUT_ROWS} rows):")
    print(f"    Total time  : {throughput_time_ms:.3f} ms")
    print(f"    Throughput  : {rows_per_sec:,.0f} rows/sec")
    print()

    return latency_median_ms, throughput_time_ms, rows_per_sec


# ──────────────────────────────────────────────
# 7. SUMMARY TABLE & JSON OUTPUT
# ──────────────────────────────────────────────
def print_summary_and_save(
    load_time, train_time, best_iter,
    auc, acc, f1, prec, rec,
    latency_ms, throughput_ms, rows_per_sec,
):
    banner("BENCHMARK SUMMARY — r5.2xlarge")

    fmt_time = lambda s: f"{s:.3f}s" if s < 60 else f"{s/60:.2f}min"

    rows = [
        ("Thời gian load data",           fmt_time(load_time)),
        ("Thời gian training",             fmt_time(train_time)),
        ("Best iteration",                 str(best_iter)),
        ("AUC-ROC",                        f"{auc:.6f}"),
        ("Accuracy",                       f"{acc:.6f}"),
        ("F1-Score",                       f"{f1:.6f}"),
        ("Precision",                      f"{prec:.6f}"),
        ("Recall",                         f"{rec:.6f}"),
        ("Inference latency (1 row)",      f"{latency_ms:.3f} ms"),
        ("Inference throughput (1000 rows)", f"{throughput_ms:.1f} ms  ({rows_per_sec:,.0f} rows/s)"),
    ]

    col_w = max(len(r[0]) for r in rows) + 2
    separator()
    print(f"  {'Metric':<{col_w}} {'Kết quả'}")
    separator()
    for metric, value in rows:
        print(f"  {metric:<{col_w}} {value}")
    separator()
    print()

    # Save JSON
    result = {
        "instance_type": "r5.2xlarge",
        "dataset": "creditcard-fraud",
        "model": "LightGBM",
        "metrics": {
            "load_time_sec": round(load_time, 4),
            "train_time_sec": round(train_time, 4),
            "best_iteration": best_iter,
            "auc_roc": round(auc, 6),
            "accuracy": round(acc, 6),
            "f1_score": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "inference_latency_1row_ms": round(latency_ms, 4),
            "inference_throughput_1000rows_ms": round(throughput_ms, 4),
            "inference_throughput_rows_per_sec": round(rows_per_sec, 2),
        },
        "lgbm_params": LGBM_PARAMS,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ✓ Results saved → {OUTPUT_PATH}")
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print_env_info()

    df, load_time               = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, train_time, best_iter = train_model(X_train, X_test, y_train, y_test)
    auc, acc, f1, prec, rec      = evaluate(model, X_test, y_test)
    latency_ms, throughput_ms, rows_per_sec = benchmark_inference(model, X_test)

    print_summary_and_save(
        load_time, train_time, best_iter,
        auc, acc, f1, prec, rec,
        latency_ms, throughput_ms, rows_per_sec,
    )


if __name__ == "__main__":
    main()
