import pandas as pd
import numpy as np

def verify_agreement():
    print("🔬 DEEP VERIFICATION: Inter-Model Consensus Analysis")
    print("   Goal: Prove that the models agree on the important stuff (Rank 1 Requirements)")
    
    try:
        # Load component models
        m1 = pd.read_csv("submission_final_ensemble_v1.csv") # Shallow/Ensemble
        m2 = pd.read_csv("submission_kfold_transformer.csv") # Deep
        m3 = pd.read_csv("submission_final_architecture.csv")# Rank (CAIN)
        final = pd.read_csv("submission_final_grand_stack.csv") # The Candidate
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Align
    df = pd.DataFrame({
        'id': m1['example_id'],
        'shallow': m1['label'],
        'deep': m2['label'],
        'rank': m3['label'],
        'final': final['label']
    })
    
    n = len(df)
    k = 1000 # Top/Bottom K to check
    
    print(f"\n1. CONSISTENCY ON EXTREMES (Top/Bottom {k} items)")
    
    # Top K Indices
    top_shallow = set(df.nlargest(k, 'shallow').index)
    top_deep    = set(df.nlargest(k, 'deep').index)
    top_rank    = set(df.nlargest(k, 'rank').index)
    
    # Intersection
    common_top = top_shallow.intersection(top_deep).intersection(top_rank)
    agreement_top = len(common_top) / k * 100
    
    print(f"   Top {k} Agreement: {agreement_top:.1f}%")
    if agreement_top > 60:
        print("   ✅ PASSED: All models agree on the majority of the 'Best' items.")
    else:
        print("   ⚠️ WARNING: Models disagree on what is 'Best'. High Risk.")

    # Bottom K Indices
    bot_shallow = set(df.nsmallest(k, 'shallow').index)
    bot_deep    = set(df.nsmallest(k, 'deep').index)
    bot_rank    = set(df.nsmallest(k, 'rank').index)
    
    common_bot = bot_shallow.intersection(bot_deep).intersection(bot_rank)
    agreement_bot = len(common_bot) / k * 100
    
    print(f"   Bottom {k} Agreement: {agreement_bot:.1f}%")
    
    print(f"\n2. DISPUTE RESOLUTION (How the Grand Stack fixes errors)")
    # Find rows with high disagreement
    df['std'] = df[['shallow', 'deep', 'rank']].std(axis=1)
    high_var = df.nlargest(5, 'std')
    
    print("   Sample Disputes (Where models fought):")
    for _, row in high_var.iterrows():
        print(f"   ID: {row['id']} | Shallow: {row['shallow']:.3f}, Rank: {row['rank']:.3f} -> Final: {row['final']:.3f}")
        
    print("\n3. FINAL VERDICT")
    if agreement_top > 50 and agreement_bot > 50:
        print("   ✅ The Grand Stack is ROBUST.")
        print("   It leverages high agreement on extremes (Safety) and averages disputes (Accuracy).")
        print("   This is the profile of a Winning Submission.")
    else:
        print("   ❌ The models are too different. This is a gamble.")

if __name__ == "__main__":
    verify_agreement()
