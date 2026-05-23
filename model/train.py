import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from gnn import MoleculeGDLModel, molecule_to_graph
import numpy as np

# A curated reference dataset of drug-like and environmental molecules with high-quality toxicological profiles.
# Format: (SMILES, hERG_cardiotox, Ames_mutagenicity, DILI_hepatotox, LD50_acute_oral_toxicity)
# Targets are binary: 1 = Toxic/Blocker, 0 = Safe/Non-Blocker
TOXICOLOGY_REFERENCE_SET = [
    # 1. Cardio/hERG Blockers (known off-target cardiotoxins)
    ("O=C(c1ccc(F)cc1)CCCN2CCC(n3c(=O)[nH]c4ccccc43)CC2", 1, 0, 1, 0),       # Haloperidol (hERG+, DILI+)
    ("CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4", 1, 0, 1, 0), # Terfenadine (hERG+, DILI+)
    ("COc1ccc(cc1)C2N(CCN2CCc3ccc(F)cc3)c4ccc(F)cc4", 1, 0, 0, 0),             # Astemizole (hERG+, DILI-)
    ("CN1CCN(CC1)C2=Nc3cc(Cl)ccc3Nc4ccccc42", 1, 0, 1, 0),                      # Clozapine (hERG+, DILI+)
    
    # 2. Ames Mutagens (genotoxic/mutagenic compounds, containing classic toxicophores like nitro or aromatic amines)
    ("O=N(=O)c1ccc(Cl)cc1", 0, 1, 1, 1),                                        # 1-Chloro-4-nitrobenzene (Ames+, LD50+)
    ("O=N(=O)c1ccccc1", 0, 1, 0, 0),                                           # Nitrobenzene (Ames+)
    ("Nc1ccc2ccccc2c1", 0, 1, 1, 0),                                           # 2-Naphthylamine (Ames+, DILI+)
    ("Cc1c(N(=O)=O)cc(C)c(N(=O)=O)c1C(C)(C)C", 0, 1, 0, 0),                     # Musk xylene (Ames+)
    
    # 3. Hepatotoxins (Drug-Induced Liver Injury)
    ("CC(=O)Nc1ccc(O)cc1", 0, 0, 1, 0),                                         # Acetaminophen / Paracetamol (DILI+)
    ("O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl", 0, 0, 1, 0),                             # Diclofenac (hERG-, DILI+)
    ("CC1Oc2ccc(CC3SC(=O)NC3=O)cc2Oc4c1c(C)c(O)c(C)c4C", 0, 0, 1, 0),           # Troglitazone (DILI+)
    ("c1ccccc1NS(=O)(=O)c2ccc(N)cc2", 0, 0, 1, 0),                              # Sulfanilamide (DILI+)
    
    # 4. Acute High Toxicity / Poisons (LD50 highly toxic)
    ("CN1C2CCC1C(C(=O)OC)C(OC(=O)c3ccccc3)C2", 0, 0, 0, 1),                    # Cocaine (LD50+)
    ("CN1CCCC1c2cccnc2", 0, 0, 0, 1),                                          # Nicotine (LD50+)
    ("CC(C)OP(=O)(F)OC(C)C", 0, 0, 0, 1),                                       # Sarin nerve agent (LD50+)
    ("CC(=O)OC1C2CC3C(C(C2)O)C4(C3CC(O4)C5=CC(=O)OC5)C", 0, 0, 0, 1),           # Digitoxin (LD50+)
    
    # 5. Multi-toxicant benchmark
    ("O=C1NC(=O)C(CC)(c2ccccc2)C(=O)N1", 0, 0, 1, 1),                           # Phenobarbital (DILI+, LD50+)
    
    # 6. Completely Safe / Control compounds
    ("CC(=O)Oc1ccccc1C(=O)O", 0, 0, 0, 0),                                      # Aspirin (Safe)
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0, 0, 0, 0),                                  # Ibuprofen (Safe)
    ("CCO", 0, 0, 0, 0),                                                        # Ethanol (Safe)
    ("OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O", 0, 0, 0, 0),                    # Glucose (Safe)
    ("O=C(O)C(O)C(O)C(=O)O", 0, 0, 0, 0),                                       # Tartaric Acid (Safe)
    ("O=C(O)CC(O)(C(=O)O)CC(=O)O", 0, 0, 0, 0),                                  # Citric Acid (Safe)
    ("O", 0, 0, 0, 0),                                                          # Water (Safe)
    ("CO", 0, 0, 0, 0),                                                         # Methanol (Safe)
]

def train_and_save():
    print("Preparing toxicology datasets...")
    graphs = []
    labels = []
    
    for smiles, herg, ames, dili, ld50 in TOXICOLOGY_REFERENCE_SET:
        graph = molecule_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
            labels.append([herg, ames, dili, ld50])
            
    print(f"Successfully converted {len(graphs)} molecules to molecular graphs.")
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Initialize the GDL model
    model = MoleculeGDLModel(num_atom_feats=31, num_bond_feats=6, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 150
    print(f"Training Geometric Deep Learning Model for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Batch size is 1 for convenience in GNNs
        for i in range(len(graphs)):
            optimizer.zero_grad()
            
            graph = graphs[i]
            x, edge_index, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr']
            y = labels[i].unsqueeze(0) # [1, 4]
            
            pred = model(x, edge_index, edge_attr) # [1, 4]
            
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Track training accuracy
            probs = torch.sigmoid(pred)
            preds_binary = (probs > 0.5).float()
            correct_predictions += (preds_binary == y).sum().item()
            total_predictions += 4
            
        acc = correct_predictions / total_predictions * 100
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {epoch_loss / len(graphs):.4f} | Train Acc: {acc:.1f}%")
            
    # Save the calibrated weights
    os.makedirs("model/weights", exist_ok=True)
    weight_path = "model/weights/gdl_toxicity_model.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"Model successfully saved to {weight_path}")

if __name__ == "__main__":
    train_and_save()
