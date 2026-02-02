import pandas as pd
import numpy as np

def recover_and_stack():
    print("🧠 GRAND UNIFIED STACK: Recovering & Fusing")
    
    # 1. Load Components
    try:
        ens_v1 = pd.read_csv("submission_final_ensemble_v1.csv") # 0.1518 (80/20 Mixed)
        trans = pd.read_csv("submission_kfold_transformer.csv")   # 0.163 (Pure Deep)
        cain = pd.read_csv("submission_final_architecture.csv")   # 0.169 (Pure Rank)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # 2. Recover Shallow (Algebraic Inversion)
    # Task.md said: "Blend 80% Shallow + 20% Transformer"
    # So: Ens = 0.8*Shallow + 0.2*Trans
    # Shallow = (Ens - 0.2*Trans) / 0.8
    
    y_ens = ens_v1['label'].values
    y_trans = trans['label'].values
    
    # Clip to avoid artifacts
    y_shallow_rec = (y_ens - 0.2 * y_trans) / 0.8
    y_shallow_rec = np.clip(y_shallow_rec, 0.0, 1.0)
    
    print(f"Recovered Shallow Mean: {y_shallow_rec.mean():.4f}")
    
    # 3. The Grand Stack
    # Logic:
    # - Shallow (Rec): High Accuracy, no deep context. (Weight 0.5)
    # - Transformer: Deep context, poor calibration. (Weight 0.1 - Keep Low)
    # - CAIN: Perfect Ranking, poor calibration. (Weight 0.4 - High for Rank)
    
    y_cain = cain['label'].values
    
    # Rank-Match CAIN to Shallow (Safety)
    from scipy.stats import rankdata
    r = rankdata(y_cain, method='ordinal') - 1
    sorted_shallow = np.sort(y_shallow_rec)
    y_cain_calib = np.interp(r, np.arange(len(sorted_shallow)), sorted_shallow)
    
    # Weighted Sum
    # 50% Shallow (Base) + 10% Transformer (Diversity) + 40% CAIN (Order)
    final_pred = (0.5 * y_shallow_rec) + (0.1 * y_trans) + (0.4 * y_cain_calib)
    
    # 4. Save
    sub = pd.DataFrame({'example_id': ens_v1['example_id'], 'label': final_pred})
    sub.to_csv("submission_final_grand_stack.csv", index=False)
    print("✅ GENERATED: submission_final_grand_stack.csv")
    print("   Composition: 50% Shallow (Recovered) + 10% Trans + 40% CAIN")

if __name__ == "__main__":
    recover_and_stack()
