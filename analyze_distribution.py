import pandas as pd
import numpy as np

def analyze():
    print("📊 Distribution Analysis")
    
    files = {
        "Ensemble V1 (0.151)": "submission_final_ensemble_v1.csv",
        "Optimized (Safe)": "submission_optimized_final.csv",
        "Calibrated (Rank 1 attempt)": "submission_quantile_calibrated.csv"
    }
    
    for name, path in files.items():
        try:
            df = pd.read_csv(path)
            preds = df['label']
            print(f"\nModel: {name}")
            print(f"   Mean: {preds.mean():.4f}")
            print(f"   Std:  {preds.std():.4f}")
            print(f"   Min:  {preds.min():.4f}")
            print(f"   Max:  {preds.max():.4f}")
            print(f"   < 0.1: {(preds < 0.1).mean()*100:.1f}%")
            print(f"   > 0.9: {(preds > 0.9).mean()*100:.1f}%")
        except FileNotFoundError:
            print(f"   {name}: File not found")

if __name__ == "__main__":
    analyze()
