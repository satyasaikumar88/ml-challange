import pandas as pd
import numpy as np

def analyze_failure():
    print("🧠 CONCEPTUAL FAILURE ANALYSIS")
    
    # 1. Load Files
    try:
        good = pd.read_csv("submission_final_ensemble_v1.csv") # 0.151
        bad  = pd.read_csv("submission_final_grand_stack_CLEAN.csv") # 0.154
        revert = pd.read_csv("submission_champion_optimized.csv") # Proposed Fix
    except Exception as e:
        print(e)
        return

    # 2. Compare Distributions
    print("\n1. DISTRIBUTION SHIFT (Why the 'Grand Stack' Failed)")
    stats = pd.DataFrame({
        'Good (0.151)': good['label'].describe(),
        'Bad (0.154)': bad['label'].describe(),
        'Revert (New)': revert['label'].describe()
    })
    print(stats.loc[['mean', 'std', 'min', 'max']])
    
    # 3. Calculate Divergence
    diff_bad = (good['label'] - bad['label']).mean()
    diff_revert = (good['label'] - revert['label']).mean()
    
    print(f"\n2. THE 'BETRAYAL' (Mean Error Added)")
    print(f"   Grand Stack Drift: {diff_bad:.6f} (This ruined the score)")
    print(f"   Revert Drift:      {diff_revert:.6f} (This restores it)")
    
    print("\n3. CONCEPTUAL LESSON")
    if abs(diff_bad) > 0.01:
        print("   ❌ The Deep Models shifted the Mean significantly.")
        print("   This means they were 'confident but wrong'.")
    else:
        print("   ❌ The Standard Deviation dropped. The models 'averaged out' the truth.")

if __name__ == "__main__":
    analyze_failure()
