import os
import sys
import gzip
import csv
import io
import urllib.request
from rdkit import Chem
import torch

# Ensure model directory is on path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.gnn import molecule_to_graph

def smiles_to_graph(smiles, y_val):
    """
    Wrapper around model.gnn.molecule_to_graph to maintain backward compatibility.
    """
    graph = molecule_to_graph(smiles)
    if graph is not None:
        graph['y'] = torch.tensor([y_val], dtype=torch.float32)
    return graph

def main():
    print("Building Geometric Graphs from ClinTox Dataset...")
    csv_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    
    try:
        # Download and decompress the clinTox file using built-in urllib and gzip
        req = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                csv_data = uncompressed.read().decode('utf-8')
                
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(reader)
        
        # Take the first 5 molecules
        sample_rows = rows[:5]
        
        graphs = []
        for idx, row in enumerate(sample_rows):
            smiles = row['smiles']
            try:
                ct_tox = float(row['CT_TOX'])
            except ValueError:
                ct_tox = 0.0
                
            graph = smiles_to_graph(smiles, ct_tox)
            if graph is not None:
                graphs.append((smiles, graph))
                
        print(f"\nSuccessfully converted {len(graphs)} molecules into Graphs.\n")
        for idx, (smiles, g) in enumerate(graphs):
            num_atoms = g['x'].size(0)
            num_bonds = g['edge_index'].size(1) // 2
            print(f"Graph [{idx}]:")
            print(f"  SMILES: {smiles}")
            print(f"  Nodes (Atoms): {num_atoms}")
            print(f"  Edges (Bonds): {num_bonds}")
            print(f"  -> Target Toxicity (y): {g['y'].item()}\n")
            
    except Exception as e:
        print(f"Failed to fetch or parse dataset: {e}")
        print("Falling back to local reference mocks...")
        # Local mock references
        presets = [
            ("CC(=O)Oc1ccccc1C(=O)O", 0.0), # Aspirin
            ("O=C(c1ccc(F)cc1)CCCN2CCC(n3c(=O)[nH]c4ccccc43)CC2", 1.0) # Haloperidol
        ]
        graphs = []
        for idx, (smiles, y_val) in enumerate(presets):
            graph = smiles_to_graph(smiles, y_val)
            if graph is not None:
                graphs.append((smiles, graph))
        
        print(f"\nSuccessfully converted {len(graphs)} fallback preset molecules into Graphs.\n")
        for idx, (smiles, g) in enumerate(graphs):
            num_atoms = g['x'].size(0)
            num_bonds = g['edge_index'].size(1) // 2
            print(f"Graph [{idx}]:")
            print(f"  SMILES: {smiles}")
            print(f"  Nodes (Atoms): {num_atoms}")
            print(f"  Edges (Bonds): {num_bonds}")
            print(f"  -> Target Toxicity (y): {g['y'].item()}\n")

if __name__ == '__main__':
    main()
