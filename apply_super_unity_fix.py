import pandas as pd
import numpy as np
import json

def restore_super_unity():
    print("🏆 RESTORING 'SUPER-UNITY' VALUES (> 1.0)")
    
    # 1. Load Data
    try:
        # Load Final File
        df = pd.read_csv("submission_final_perfected.csv")
        y_pred = df['label'].values
        
        # Load Ground Truth
        with open("train.jsonl", "r", encoding="utf-8") as f:
            y_train = np.array([json.loads(line)['label'] for line in f])
            
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Check Constraints
    train_max = y_train.max()
    pred_max = y_pred.max()
    
    print(f"Ground Truth Max: {train_max:.5f}")
    print(f"Current Pred Max: {pred_max:.5f}")
    
    if train_max <= 1.01:
        print("Wait, train max is normal. No fix needed.")
        return

    # 3. Apply Explicit Quantile Matching for High End
    print("Applying High-End Correction...")
    
    # We only care about the top 1% that should be > 1.0
    # Let's do a full re-calibration just to be safe, using the exact train distribution
    
    y_train_sorted = np.sort(y_train)
    rank = np.argsort(np.argsort(y_pred))
    n_test = len(y_pred)
    n_train = len(y_train)
    
    y_target = np.interp(
        np.linspace(0, 1, n_test),
        np.linspace(0, 1, n_train),
        y_train_sorted
    )
    
    y_fixed = y_target[rank]
    
    print(f"New Pred Max: {y_fixed.max():.5f}")
    
    if y_fixed.max() > 1.01:
        print("✅ SUCCESS: Super-Unity values restored.")
        df['label'] = y_fixed
        df.to_csv("submission_final_perfected.csv", index=False)
        print("Overwrote submission_final_perfected.csv")
    else:
        print("❌ FAILED: Interpolation didn't grab the max.")

if __name__ == "__main__":
    restore_super_unity()
