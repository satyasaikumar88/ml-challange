import pandas as pd
import numpy as np

def check_max():
    print("WINNER'S EDGE CHECK: Super-Unity Values")
    
    # 1. Load Files
    df = pd.read_csv("submission_final_perfected.csv")
    
    max_val = df['label'].max()
    min_val = df['label'].min()
    
    print(f"Submission Max: {max_val:.5f}")
    
    if max_val > 1.01:
        print("SUCCESS: We are predicting values > 1.0 (matching ground truth 1.06).")
        print("   Most users clip at 1.0. This gives us the edge.")
    else:
        print("WARNING: Predictions are capped at 1.0.")
        print("   We are missing the 'Super-Unity' outliers.")

if __name__ == "__main__":
    check_max()
