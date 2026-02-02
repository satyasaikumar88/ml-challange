import pandas as pd
import numpy as np

def compare_safety():
    print("SAFETY AUDIT: Champion vs Best (0.15183)")
    
    # 1. Load Files
    try:
        df_best = pd.read_csv("submission_final_ensemble_v1.csv")
        df_champ = pd.read_csv("submission_final_champion.csv")
    except:
        print("Error loading files.")
        return

    # 2. Key Metrics
    y_best = df_best['label'].values
    y_champ = df_champ['label'].values
    
    # Correction: Use corr() method
    correlation = pd.Series(y_best).corr(pd.Series(y_champ))
    mae_diff = np.mean(np.abs(y_best - y_champ))
    
    print(f"Correlation: {correlation:.6f} (Should be > 0.99)")
    print(f"Avg Change:  {mae_diff:.6f} (Should be < 0.02)")
    
    # 3. Why it is safe
    if correlation > 0.99:
        print("VERDICT: SAFE. The logic is identical to your Best Score.")
    else:
        print("VERDICT: RISKY. The logic has changed significantly.")
        
    # 4. Range Check
    print(f"Best Range: {y_best.min():.4f} - {y_best.max():.4f}")
    print(f"Champ Range: {y_champ.min():.4f} - {y_champ.max():.4f}")

if __name__ == "__main__":
    compare_safety()
