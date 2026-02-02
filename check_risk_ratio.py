import pandas as pd
import numpy as np

def risk_check():
    print("⚖️ DEVIL'S ADVOCATE CHECK: Risk Ratio")
    
    df_old = pd.read_csv("submission_final_ensemble_v1.csv")
    df_new = pd.read_csv("submission_final_perfected.csv")
    
    # Calculate Diff
    diff = np.abs(df_old['label'] - df_new['label'])
    
    # Count "Big Moves" (> 0.2 change)
    # A change of 0.2 is massive (e.g. 0.6 -> 0.8).
    # If we have too many of these, we are gambling.
    
    big_moves = (diff > 0.2).sum()
    total = len(diff)
    ratio = big_moves / total * 100
    
    print(f"Total Predictions: {total}")
    print(f"Big Moves (>0.2):  {big_moves}")
    print(f"Risk Ratio:        {ratio:.2f}%")
    
    print("\n--- HONEST VERDICT ---")
    if ratio > 5.0:
        print("⚠️ HIGH RISK. I changed >5% of your answers drastically.")
        print("   This IS a gamble.")
    elif ratio < 0.1:
        print("❌ TOO SAFE. I barely changed anything.")
        print("   This won't win.")
    else:
        print("✅ CALCULATED RISK. (0.1% - 5%)")
        print("   I changed enough to win, but not enough to ruin you.")
        print("   This is Engineering, not Gambling.")

if __name__ == "__main__":
    risk_check()
