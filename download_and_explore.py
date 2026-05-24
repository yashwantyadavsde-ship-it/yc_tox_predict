import os
import gzip
import csv
import io
import urllib.request

def main():
    print("Welcome to the Toxicity Prediction Pipeline!")
    print("Downloading and preparing the ClinTox dataset...\n")

    # The ClinTox dataset contains:
    # - FDA_APPROVED: 1 if FDA approved, 0 otherwise
    # - FDA_Tox: 1 if failed clinical trials due to toxicity, 0 otherwise
    
    csv_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    
    print(f"Fetching from {csv_url} ...")
    
    try:
        req = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                csv_data = uncompressed.read().decode('utf-8')
                
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(reader)
        
        print(f"\nTotal number of compounds: {len(rows)}")
        
        # Display sample
        print("\nSample Data (SMILES | FDA_APPROVED | FDA_Tox):")
        for i in range(min(5, len(rows))):
            row = rows[i]
            print(f"[{i}] SMILES: {row['smiles']} => FDA_APPROVED: {row['FDA_APPROVED']}, CT_TOX: {row['CT_TOX']}")

        # Count toxic drugs
        toxic_count = sum(1 for row in rows if row.get('CT_TOX') == '1')
        safe_count = len(rows) - toxic_count
        
        print(f"\nDataset balance for FDA_Tox (Clinical Trial Failure due to toxicity):")
        print(f"- Safe (0): {safe_count}")
        print(f"- Toxic (1): {toxic_count}")

        print("\nData loading and exploration completed successfully!")
        
    except Exception as e:
        print(f"Failed to fetch or parse dataset: {e}")
        print("Please verify your internet connection or try again later.")

if __name__ == '__main__':
    main()
