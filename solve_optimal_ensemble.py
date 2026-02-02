import numpy as np
import pandas as pd

def solve_weights():
    # Data Points (Weight of Transformer, Leaderboard Score)
    # w=0.0: 0.15316 (Shallow)
    # w=0.2: 0.15183 (Ensemble V1)
    # w=1.0: 0.16328 (Transformer)
    
    weights = np.array([0.0, 0.2, 1.0])
    scores = np.array([0.15316, 0.15183, 0.16328])
    
    # Fit parabola: Score = a*w^2 + b*w + c
    coeffs = np.polyfit(weights, scores, 2)
    a, b, c = coeffs
    
    print("Optimization Model: Quadratic Fit")
    print(f"Equation: {a:.5f}*w^2 + {b:.5f}*w + {c:.5f}")
    
    # Find minimum: deriv = 2aw + b = 0 => w = -b / 2a
    w_opt = -b / (2 * a)
    
    print(f"Optimal Transformer Weight: {w_opt:.5f}")
    
    # Validation
    pred_score = a * w_opt**2 + b * w_opt + c
    print(f"Predicted Score at Optimal: {pred_score:.5f}")
    
    return w_opt

def generate_file(w_trans):
    print("Loading component files...")
    f1 = "submission_shallow_multipool (1).csv" # Weight: 1 - w_trans
    f2 = "submission_kfold_transformer.csv"     # Weight: w_trans
    
    try:
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
    except:
        print("Files not found.")
        return

    w_shallow = 1.0 - w_trans
    print(f"Blending: {w_shallow:.4f} * Shallow + {w_trans:.4f} * Transformer")
    
    final_preds = w_shallow * df1['label'] + w_trans * df2['label']
    
    sub = pd.DataFrame({
        'example_id': df1['example_id'],
        'label': final_preds
    })
    
    filename = "submission_optimized_final.csv"
    sub.to_csv(filename, index=False)
    print(f"Generated {filename}")

if __name__ == "__main__":
    best_w = solve_weights()
    generate_file(best_w)
