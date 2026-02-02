import pandas as pd
import numpy as np

def verify_ensemble():
    print("Starting Independent Verification...")
    
    # 1. Load the inputs used
    try:
        df_shallow = pd.read_csv("submission_shallow_multipool (1).csv")
        print(f"Input 1 (Shallow): Loaded {len(df_shallow)} rows (Score ~0.153)")
    except Exception as e:
        print(f"Input 1 Failed: {e}")
        return

    try:
        df_trans = pd.read_csv("submission_kfold_transformer.csv")
        print(f"Input 2 (Transformer): Loaded {len(df_trans)} rows (Score ~0.163)")
    except Exception as e:
        print(f"Input 2 Failed: {e}")
        return

    # 2. Load the output file to check
    try:
        df_final = pd.read_csv("submission_final_ensemble_v1.csv")
        print(f"Output (Final): Loaded {len(df_final)} rows")
    except Exception as e:
        print(f"Output Failed: {e}")
        return

    # 3. Check IDs match perfectly
    if not df_shallow['example_id'].equals(df_final['example_id']):
        print("CRITICAL ERROR: Output IDs do not match Input IDs!")
        return
    if not df_trans['example_id'].equals(df_final['example_id']):
        print("CRITICAL ERROR: Transformer IDs do not match Output IDs!")
        return
    print("ID Alignment: Perfect")

    # 4. Verify the Math
    # Formula: 0.8 * Shallow + 0.2 * Transformer
    expected = (0.8 * df_shallow['label']) + (0.2 * df_trans['label'])
    expected = expected.clip(0, 1) # Although clip was at end, logic holds.
    
    # Calculate difference
    diff = np.abs(df_final['label'] - expected)
    max_diff = diff.max()
    
    print(f"\nMath Verification (0.8 * A + 0.2 * B):")
    print(f"   Max Difference found: {max_diff:.9f}")
    
    if max_diff < 1e-9:
        print("MATH IS EXACT. The file contains exactly what was promised.")
    else:
        print("MATH MISMATCH. Something is wrong.")

    # 5. Show samples
    print("\nSample Rows:")
    print(f"{'ID':<12} | {'Shallow':<10} | {'Trans':<10} | {'Final':<10} | {'Calc check':<10}")
    print("-" * 65)
    for i in range(5):
        row_id = df_final.iloc[i]['example_id']
        s_val = df_shallow.iloc[i]['label']
        t_val = df_trans.iloc[i]['label']
        f_val = df_final.iloc[i]['label']
        calc = (0.8 * s_val) + (0.2 * t_val)
        print(f"{row_id:<12} | {s_val:.4f}     | {t_val:.4f}     | {f_val:.4f}     | {calc:.4f}")

if __name__ == "__main__":
    verify_ensemble()
