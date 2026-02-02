import pandas as pd
import numpy as np
import json

def analyze_segments():
    print("GRANDMASTER CHECK: Segmented Bias Analysis")
    
    # 1. Load Train Data & Calculate Lengths
    print("Loading Train Data...")
    train_lens = []
    train_y = []
    with open("train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            train_lens.append(len(data['input_ids']))
            train_y.append(data['label'])
            
    df_train = pd.DataFrame({'len': train_lens, 'label': train_y})
    
    # 2. Define Segments (Short, Medium, Long)
    # Use quantiles to split equally
    q1 = df_train['len'].quantile(0.33)
    q2 = df_train['len'].quantile(0.66)
    
    print(f"Segments for split: {q1} | {q2}")
    
    # Calculate Train Means
    means = df_train.groupby(pd.cut(df_train['len'], [-1, q1, q2, 10000])).agg({'label': ['mean', 'std', 'count']})
    print("\n--- Training Data Segments ---")
    print(means)
    
    # 3. Check Current Submission Behavior
    print("\nLoading Submission...")
    df_sub = pd.read_csv("submission_final_perfected.csv")
    
    # We need Test Lengths to map them
    test_lens = []
    with open("test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            test_lens.append(len(data['input_ids']))
            
    df_sub['len'] = test_lens
    
    # Calculate Submission Means
    sub_means = df_sub.groupby(pd.cut(df_sub['len'], [-1, q1, q2, 10000])).agg({'label': ['mean', 'std', 'count']})
    print("\n--- Submission Segments ---")
    print(sub_means)
    
    # 4. Compare
    print("\n--- VERDICT ---")
    # Extract means
    train_m = means['label']['mean'].values
    sub_m = sub_means['label']['mean'].values
    
    diffs = np.abs(train_m - sub_m)
    max_diff = np.max(diffs)
    
    print(f"Max Segment Divergence: {max_diff:.4f}")
    
    if max_diff > 0.05:
        print("CRITICAL: The submission ignores segment bias!")
        print("   Action: MUST run Segmented Calibration.")
    else:
        print("PERFECT: The submission naturally respects segment bias.")
        print("   Global calibration was sufficient.")

if __name__ == "__main__":
    analyze_segments()
