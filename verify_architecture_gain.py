import pandas as pd
import numpy as np

def verify_gain():
    print("🔬 ARCHITECTURE AUDIT: CAIN Model vs Baseline")
    
    try:
        df_base = pd.read_csv("submission_final_ensemble_v1.csv")
        df_new = pd.read_csv("submission_final_architecture.csv")
    except:
        print("⚠️ Waiting for submission_final_architecture.csv...")
        return

    y_base = df_base['label'].values
    y_new = df_new['label'].values
    
    # 1. Correlation (lower is better for ensemble potential, high for stability)
    corr = pd.Series(y_base).corr(pd.Series(y_new))
    print(f"Correlation: {corr:.6f}")
    
    if corr < 0.95:
        print("✅ SUCCESS: The Architecture learned NEW patterns (Distinct Signal).")
    elif corr > 0.99:
        print("⚠️ WARNING: The Architecture converged to the same pattern as Baseline.")
    else:
        print("ℹ️ NOTE: Moderate similarity.")
        
    # 2. Ranking Change involved
    # How many pairs flipped order?
    # Sample 1000 pairs
    np.random.seed(42)
    idx1 = np.random.randint(0, len(y_base), 10000)
    idx2 = np.random.randint(0, len(y_base), 10000)
    
    base_diff = y_base[idx1] - y_base[idx2]
    new_diff = y_new[idx1] - y_new[idx2]
    
    # Sign match?
    agreement = np.mean(np.sign(base_diff) == np.sign(new_diff))
    print(f"Pairwise Visual Agreement: {agreement:.4f}")
    
    # 3. Range
    print(f"New Model Range: {y_new.min():.4f} - {y_new.max():.4f}")

if __name__ == "__main__":
    verify_gain()
