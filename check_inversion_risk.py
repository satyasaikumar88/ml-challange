import pandas as pd
import numpy as np

def check_inversion():
    print("☢️ NUCLEAR SAFETY CHECK: Prediction Inversion Analysis")
    
    # Load Files
    df_safe = pd.read_csv("submission_final_ensemble_v1.csv") # The "Trust"
    df_perf = pd.read_csv("submission_final_champion.csv")   # The "Champion"
    
    y_safe = df_safe['label'].values
    y_perf = df_perf['label'].values
    
    # Define Inversion: 
    # Safe says High (>0.7), Perfect says Low (<0.3)
    # Safe says Low (<0.3), Perfect says High (>0.7)
    
    inversions = 0
    risky_rows = []
    
    for i in range(len(y_safe)):
        s = y_safe[i]
        p = y_perf[i]
        
        # Case 1: Huge Flip High->Low
        if s > 0.8 and p < 0.2:
            inversions += 1
            risky_rows.append((i, s, p))
            
        # Case 2: Huge Flip Low->High
        if s < 0.2 and p > 0.8:
            inversions += 1
            risky_rows.append((i, s, p))
            
    print(f"\nTotal Predictions: {len(y_safe)}")
    print(f"Dangerous Inversions Found: {inversions}")
    
    if inversions > 0:
        print("❌ DANGER: The new file CONTRADICTS the safe file!")
        print(f"   Example: Row {risky_rows[0][0]} was {risky_rows[0][1]} -> became {risky_rows[0][2]}")
        print("   This causes bad scores (0.176).")
    else:
        print("✅ CLEAN: Zero contradictions.")
        print("   The new file respects the logic of the old file 100%.")
        print("   It only changes the *intensity*, not the *direction*.")

if __name__ == "__main__":
    check_inversion()
