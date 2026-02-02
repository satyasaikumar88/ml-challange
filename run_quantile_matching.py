import pandas as pd
import numpy as np
import json

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def quantile_match():
    print("🧠 Intelligent Mode: Distribution Calibration (Quantile Matching)")
    
    # 1. Load Ground Truth (Training Targets)
    print("1. Loading Training Data (Ground Truth)...")
    try:
        train_data = load_jsonl("train.jsonl")
        y_train = np.array([x['label'] for x in train_data])
        print(f"   Loaded {len(y_train)} labels.")
        print(f"   Train Dist -> Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    except FileNotFoundError:
        print("❌ Error: train.jsonl not found. Cannot calibrate.")
        return

    # 2. Load Best Submission
    print("2. Loading Best Prediction (Optimized File)...")
    try:
        df_sub = pd.read_csv("submission_optimized_final.csv")
        y_pred = df_sub['label'].values
        print(f"   Loaded {len(y_pred)} predictions.")
        print(f"   Test Dist  -> Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
    except FileNotFoundError:
        print("❌ Error: submission_optimized_final.csv not found.")
        return

    # 3. Perform Quantile Matching
    # Concept: Map the rank of prediction to the value in ground truth.
    # If prediction is 90th percentile in Test, give it the 90th percentile value from Train.
    print("3. Applying Quantile Matching...")
    
    # Sort references
    y_train_sorted = np.sort(y_train)
    
    # Get rank of predictions (0 to 1)
    # argsort twice gives rank
    rank = np.argsort(np.argsort(y_pred))
    
    # Map rank to index in train
    # Scaled index = rank * (n_train / n_test)
    n_train = len(y_train)
    n_test = len(y_pred)
    
    calibrated_preds = np.zeros(n_test)
    
    # Interpolation for exact mapping
    # We want values from y_train that correspond to the quantiles of y_pred
    from scipy.interpolate import interp1d
    
    # Quantiles of train
    q_train = np.linspace(0, 1, n_train)
    y_train_sorted = np.sort(y_train)
    
    # Quantiles of test predictions
    # We replace value at quantile X with value at quantile X in train
    
    # Simple approach: Rank Matching
    # 1. Rank predictions
    # 2. Assign values from train distribution based on rank
    # Note: If n_test != n_train, we interpolate the train distribution
    
    y_target = np.interp(
        np.linspace(0, 1, n_test),
        np.linspace(0, 1, n_train),
        y_train_sorted
    )
    
    # Reorder to match original prediction order
    # sorted_indices = np.argsort(y_pred)
    # calibrated_preds[sorted_indices] = y_target
    
    # BETTER: Using rank
    # rank[i] is the rank of the i-th prediction (0 to N-1)
    # We assign the element at that rank from the (interpolated) target distribution
    calibrated_preds = y_target[rank]
    
    print(f"   Calibrated -> Mean: {calibrated_preds.mean():.4f}, Std: {calibrated_preds.std():.4f}")
    
    # 4. Save
    df_sub['label'] = calibrated_preds
    filename = "submission_quantile_calibrated.csv"
    df_sub.to_csv(filename, index=False)
    print(f"🚀 Success! Generated {filename}")
    print("This file has the EXACT statistical distribution of the training set.")

if __name__ == "__main__":
    quantile_match()
