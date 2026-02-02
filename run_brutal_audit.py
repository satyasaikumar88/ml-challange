import pandas as pd
import numpy as np

def brutal_audit():
    print("💀 RUTHLESS POST-MORTEM AUDIT")
    
    # Load Files
    try:
        best = pd.read_csv("submission_final_ensemble_v1.csv") # 0.1518
        worst = pd.read_csv("submission_final_grand_stack_CLEAN.csv") # 0.1548
        rank1_target = 0.14631
    except Exception as e:
        print(f"File Error: {e}")
        return

    print(f"\n1. FAILURE QUANTIFICATION")
    print(f"   Target: {rank1_target}")
    print(f"   Best:   0.15183 (+{0.15183 - rank1_target:.5f}) -> BAD")
    print(f"   Worst:  0.15481 (+{0.15481 - rank1_target:.5f}) -> UNACCEPTABLE")
    print(f"   REGRESSION: +{0.15481 - 0.15183:.5f} (The experiment hurt us)")
    
    print(f"\n2. VARIANCE COLLAPSE (The Root Cause)")
    std_best = best['label'].std()
    std_worst = worst['label'].std()
    print(f"   Best StdDev:  {std_best:.6f} (Sharper)")
    print(f"   Worst StdDev: {std_worst:.6f} (Blurred)")
    
    if std_worst < std_best:
        print("   ❌ DIAGNOSIS: Over-Smoothing. The ensemble averaged out the signal.")
    
    print(f"\n3. CORRELATION CHECK (Why Stacking Failed)")
    corr = best['label'].corr(worst['label'])
    print(f"   Correlation: {corr:.6f}")
    if corr > 0.95:
        print("   ❌ DIAGNOSIS: High Correlation. You added complexity without adding diversity.")
        
    print(f"\n4. RANGE COMPRESSION")
    print(f"   Best Range:  {best['label'].min():.4f} - {best['label'].max():.4f}")
    print(f"   Worst Range: {worst['label'].min():.4f} - {worst['label'].max():.4f}")

if __name__ == "__main__":
    brutal_audit()
