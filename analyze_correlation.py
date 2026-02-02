import pandas as pd
import numpy as np

def solve_optimal_weight():
    print("🔬 Scientific Weight Optimization")
    
    # 1. Load Data
    try:
        df1 = pd.read_csv("submission_shallow_multipool (1).csv") # Score 0.153
        df2 = pd.read_csv("submission_kfold_transformer.csv")     # Score 0.163
        print("✅ Loaded both submission files.")
    except FileNotFoundError:
        print("❌ Error: Files not found.")
        return

    preds1 = df1['label'].values
    preds2 = df2['label'].values

    # 2. Correlation Analysis
    corr = np.corrcoef(preds1, preds2)[0, 1]
    print(f"\n📊 Correlation (r): {corr:.4f}")
    
    if corr > 0.95:
        print("   -> Models are highly correlated. Gain will be small.")
    else:
        print("   -> Models are diverse! Significant gain expected.")

    # 3. Optimal Weight Calculation
    # Theory: Minimize Variance of (w*M1 + (1-w)*M2)
    # Optimal w = (Var2 - Cov) / (Var1 + Var2 - 2*Cov)
    # We use Error Variance ~ Score^2
    
    s1 = 0.15316  # Score of Model 1 (Shallow)
    s2 = 0.16328  # Score of Model 2 (Transformer)
    
    # Estimated Error Variance (Simulated)
    # Covariance = r * s1 * s2
    cov = corr * s1 * s2
    v1 = s1 ** 2
    v2 = s2 ** 2
    
    # Formula for weight of Model 1
    w_opt = (v2 - cov) / (v1 + v2 - 2*cov)
    
    # Clamp between 0 and 1
    w_opt = np.clip(w_opt, 0, 1)
    
    print("\n🧮 Optimization Results:")
    print(f"   Model 1 (Shallow, 0.153):   {w_opt:.4f} ({w_opt*100:.1f}%)")
    print(f"   Model 2 (Transf, 0.163):    {1-w_opt:.4f} ({(1-w_opt)*100:.1f}%)")
    
    return w_opt

if __name__ == "__main__":
    solve_optimal_weight()
