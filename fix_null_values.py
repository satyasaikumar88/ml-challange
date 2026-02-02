import pandas as pd
import numpy as np

def fix_nans():
    print("🚑 EMERGENCY NAN FIX")
    
    # Load the broken file
    try:
        df = pd.read_csv("submission_final_grand_stack.csv")
    except Exception as e:
        print(f"Could not load file: {e}")
        return

    # Check for NaNs
    nans = df[df['label'].isnull()]
    print(f"❌ FOUND {len(nans)} NULL VALUES!")
    print(nans)
    
    if len(nans) > 0:
        # Load Safe Backup
        print("Loading Safe Backup (0.151)...")
        safe = pd.read_csv("submission_final_ensemble_v1.csv")
        
        # Fill NaNs with Safe Values
        for idx in nans.index:
            eid = df.loc[idx, 'example_id']
            # Get safe value
            safe_val = safe.loc[safe['example_id'] == eid, 'label'].values
            if len(safe_val) > 0:
                print(f"   Refilling {eid} with safe value: {safe_val[0]}")
                df.loc[idx, 'label'] = safe_val[0]
            else:
                # Fallback to mean if ID missing (unlikely)
                print(f"   ⚠️ ID {eid} not in safe file! Refilling with 0.5")
                df.loc[idx, 'label'] = 0.5
                
    # Final Check
    if df['label'].isnull().sum() == 0:
        print("✅ ALL NULLS ELIMINATED.")
        df.to_csv("submission_final_grand_stack_FIXED.csv", index=False)
        print("🚀 SAVED: submission_final_grand_stack_FIXED.csv")
    else:
        print("❌ CRITICAL: NaNs still persist.")

if __name__ == "__main__":
    fix_nans()
