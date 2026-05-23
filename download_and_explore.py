import os
import pandas as pd

def main():
    print("Welcome to the Toxicity Prediction Pipeline!")
    print("Downloading and preparing the ClinTox dataset...\n")

    # The ClinTox dataset contains:
    # - FDA_APPROVED: 1 if FDA approved, 0 otherwise
    # - FDA_Tox: 1 if failed clinical trials due to toxicity, 0 otherwise
    # We will use pandas to load it directly for exploration.
    
    csv_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    
    print(f"Fetching from {csv_url} ...")
    df = pd.read_csv(csv_url)
    
    print(f"\nTotal number of compounds: {len(df)}")
    
    # The dataset contains SMILES, and the two tasks.
    print("\nSample Data (SMILES | FDA_APPROVED | FDA_Tox):")
    for i in range(5):
        row = df.iloc[i]
        print(f"[{i}] SMILES: {row['smiles']} => FDA_APPROVED: {row['FDA_APPROVED']}, CT_TOX: {row['CT_TOX']}")

    # Let's count how many toxic drugs there are
    toxic_count = df['CT_TOX'].sum()
    safe_count = len(df) - toxic_count
    
    print(f"\nDataset balance for FDA_Tox (Clinical Trial Failure due to toxicity):")
    print(f"- Safe (0): {int(safe_count)}")
    print(f"- Toxic (1): {int(toxic_count)}")

    print("\nData loading and exploration completed successfully!")

if __name__ == '__main__':
    main()
