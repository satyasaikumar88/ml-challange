import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def deep_research():
    print("DEEP RESEARCH: VALIDATION & LEAKAGE HUNT")
    
    # 1. LOAD DATA
    print("1. Loading Data...")
    train_data = load_jsonl("train.jsonl")
    test_data = load_jsonl("test.jsonl")
    
    y = np.array([x['label'] for x in train_data])
    X_seq = [tuple(x['input_ids']) for x in train_data]
    
    # 2. LEAKAGE HUNT (Exact Match Search)
    print("\n2. Hunting for Leakage (Exact Duplicates)...")
    # Build distinct train lookup
    train_lookup = {}
    for r in train_data:
        t = tuple(r['input_ids'])
        train_lookup[t] = r['label']
        
    hits = 0
    perfect_matches = []
    
    for r in test_data:
        t = tuple(r['input_ids'])
        if t in train_lookup:
            hits += 1
            perfect_matches.append((r['example_id'], train_lookup[t]))
            
    print(f"   Found {hits} Test items that exist in Train.")
    if hits > 0:
        print("   LEAKAGE DETECTED! We can use exact ground truth for these!")
        # We should overwrite these in the submission
    else:
        print("   No data leakage found. Models must generalize.")

    # 3. CALIBRATION SIMULATION (The "Trust" Check)
    print("\n3. Simulating Calibration Strategy (Training Data Split)...")
    # We simulate the exact scenario: Model is 'conservative' (under-predicts extremes)
    
    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    
    # Simulate our Rank 6 Model's behavior
    # It has 0.151 error and std dev 0.27 (vs 0.32 real)
    # We can simulate this by shrinking y_val towards the mean
    mean_y = np.mean(y_train)
    y_val_simulated = mean_y + 0.85 * (y_val - mean_y) + np.random.normal(0, 0.05, size=len(y_val))
    y_val_simulated = np.clip(y_val_simulated, 0, 1)
    
    mae_before = np.mean(np.abs(y_val - y_val_simulated))
    std_before = np.std(y_val_simulated)
    # print(f"   [Simulation] Rank 6 Model Error (MAE): {mae_before:.5f}")
    # print(f"   [Simulation] Rank 6 Std Dev: {std_before:.4f} (Too narrow!)")
    
    # Apply Quantile Calibration to the Simulated Predictions
    print("   -> Applying Quantile Matching to Simulation...")
    
    # Sort Train (Reference)
    y_train_sorted = np.sort(y_train)
    
    # Sort Simulation (Preds)
    # Get ranks
    rank = np.argsort(np.argsort(y_val_simulated))
    
    n_val = len(y_val)
    n_train = len(y_train)
    
    # Interpolate from Train Distribution
    y_target_dist = np.interp(
        np.linspace(0, 1, n_val),
        np.linspace(0, 1, n_train),
        y_train_sorted
    )
    
    y_calibrated = y_target_dist[rank]
    
    mae_after = np.mean(np.abs(y_val - y_calibrated))
    std_after = np.std(y_calibrated)
    
    # print(f"   [Result] Calibrated MAE: {mae_after:.5f}")
    # print(f"   [Result] Calibrated Std Dev: {std_after:.4f} (Restored!)")
    
    improvement = mae_before - mae_after
    
    # FINAL REPORT WRITING
    max_matches = hits
    
    with open("final_report_short.txt", "w") as f:
        f.write(f"LEAKAGE_COUNT:{max_matches}\n")
        f.write(f"SIM_MAE_BEFORE:{mae_before:.5f}\n")
        f.write(f"SIM_MAE_AFTER:{mae_after:.5f}\n")
        f.write(f"SIM_IMPROVEMENT:{improvement:.5f}\n")
        
    print("DONE writing final_report_short.txt")

if __name__ == "__main__":
    deep_research()
