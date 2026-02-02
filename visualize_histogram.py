import pandas as pd
import numpy as np
import json

def ascii_hist(data, name):
    counts, bins = np.histogram(data, bins=10, range=(0, 1))
    total = len(data)
    print(f"\nHistogram: {name}")
    print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    print("-" * 40)
    for i in range(10):
        low, high = bins[i], bins[i+1]
        c = counts[i]
        bar_len = int((c / total) * 40)
        bar = "█" * bar_len
        print(f"{low:.1f}-{high:.1f} | {bar} ({c/total*100:.1f}%)")

def visualize():
    print("👀 Visual Inspection of Distributions")
    
    # 1. Ground Truth
    with open("train.jsonl", "r", encoding="utf-8") as f:
        y_train = np.array([json.loads(line)['label'] for line in f])
    ascii_hist(y_train, "Ground Truth (Training)")

    # 2. Rank 6 (Safe)
    try:
        y_safe = pd.read_csv("submission_final_ensemble_v1.csv")['label'].values
        ascii_hist(y_safe, "Rank 6 File (Previous)")
    except: pass

    # 3. Calibrated (Winner)
    try:
        y_cal = pd.read_csv("submission_quantile_calibrated.csv")['label'].values
        ascii_hist(y_cal, "Calibrated File (Final)")
    except: pass

if __name__ == "__main__":
    visualize()
