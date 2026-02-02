import pandas as pd

def fix_notation():
    print("🔧 FIXING SCIENTIFIC NOTATION RISK")
    filename = "submission_final_perfected.csv"
    
    # Load
    df = pd.read_csv(filename)
    
    # Force formatting to 6 decimal places (standard float)
    # This prevents 1e-05
    df['label'] = df['label'].apply(lambda x: f"{x:.6f}")
    
    # Save
    df.to_csv(filename, index=False)
    print(f"✅ Re-saved {filename} with strict '0.000000' formatting.")
    print("   No more scientific notation.")

if __name__ == "__main__":
    fix_notation()
