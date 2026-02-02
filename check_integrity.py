import pandas as pd
import numpy as np

def check_integrity():
    filename = "submission_quantile_calibrated.csv"
    print(f"🔒 Final Integrity Check: {filename}")
    
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"❌ FAIL: Could not read file. {e}")
        return

    # 1. Row Count
    expected_rows = 19954 # Approx, generic check
    if len(df) < 1000:
        print(f"❌ FAIL: Only {len(df)} rows!")
        return
    print(f"✅ Rows: {len(df)}")

    # 2. NaNs
    nans = df.isnull().sum().sum()
    if nans > 0:
        print(f"❌ FAIL: Found {nans} NaN values!")
        return
    print("✅ NaNs: 0")

    # 3. Range
    min_val = df['label'].min()
    max_val = df['label'].max()
    if min_val < 0 or max_val > 1:
        print(f"❌ FAIL: Values out of range [{min_val}, {max_val}]")
        return
    print(f"✅ Range: [{min_val:.4f}, {max_val:.4f}]")

    # 4. Rank Preservation Check
    # Load original ensemble to make sure we didn't scramble the order
    try:
        df_orig = pd.read_csv("submission_final_ensemble_v1.csv")
        # Correlation should be 1.0 (Spearman/Rank)
        corr = df['label'].corr(df_orig['label'], method='spearman')
        print(f"✅ Rank Integrity (Correlation): {corr:.6f}")
        if corr < 0.999:
            print("❌ WARNING: Ranks changed! Logic error.")
        else:
            print("✅ Logic: Ranks preserved perfectly.")
    except:
        print("Could not load original for rank check.")

if __name__ == "__main__":
    check_integrity()
