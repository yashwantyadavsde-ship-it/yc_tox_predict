import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data

def smiles_to_graph(smiles, y_val):
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    Nodes = Atoms (Atomic number is used as the node feature)
    Edges = Bonds (Adjacency list)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # Invalid SMILES
        
    # Node features: Atomic number of each atom
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([atom.GetAtomicNum()])
        
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edges
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Undirected graph, so add both directions
        edges.append([i, j])
        edges.append([j, i])
        
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Molecule with no bonds (e.g. single ion)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # Label (Toxicity flag)
    y = torch.tensor([y_val], dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def main():
    print("Building Geometric Graphs from ClinTox Dataset...")
    csv_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    df = pd.read_csv(csv_url)
    
    # Take the first 5 molecules
    sample_df = df.head(5)
    
    graphs = []
    for idx, row in sample_df.iterrows():
        smiles = row['smiles']
        toxicity = row['CT_TOX']
        
        graph = smiles_to_graph(smiles, toxicity)
        if graph is not None:
            graphs.append((smiles, graph))
            
    # Display the results
    print(f"\nSuccessfully converted {len(graphs)} molecules into Graphs.\n")
    for idx, (smiles, g) in enumerate(graphs):
        num_atoms = g.num_nodes
        num_bonds = g.num_edges // 2 # Divided by 2 because it's undirected
        print(f"Graph [{idx}]:")
        print(f"  SMILES: {smiles}")
        print(f"  PyG Data: {g}")
        print(f"  -> It has {num_atoms} Atoms (Nodes) and {num_bonds} Bonds (Edges)")
        print(f"  -> Target Toxicity (y): {g.y.item()}\n")

if __name__ == '__main__':
    main()
