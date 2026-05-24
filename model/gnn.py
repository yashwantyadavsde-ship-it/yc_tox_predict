import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# ==========================================
# 1. Molecular Graph Featurization
# ==========================================

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    """
    Generate atom features:
    - Atomic number (one-hot: C, N, O, F, P, S, Cl, Br, I, Unknown)
    - Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2, Unknown)
    - Degree (one-hot: 0, 1, 2, 3, 4, 5, 6+)
    - Formal charge (integer)
    - Number of implicit Hs (one-hot: 0, 1, 2, 3, 4+)
    - Aromaticity (0 or 1)
    - Is in ring (0 or 1)
    """
    allowable_elements = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]
    degrees = [0, 1, 2, 3, 4, 5, 6]
    implicit_hs = [0, 1, 2, 3, 4]

    feats = []
    # Element one-hot
    feats.extend(one_of_k_encoding_unk(atom.GetSymbol(), allowable_elements))
    # Hybridization one-hot
    feats.extend(one_of_k_encoding_unk(atom.GetHybridization(), hybridization_types))
    # Degree one-hot
    feats.extend(one_of_k_encoding_unk(atom.GetDegree(), degrees))
    # Implicit Hs one-hot
    feats.extend(one_of_k_encoding_unk(atom.GetImplicitValence(), implicit_hs))
    # Formal charge
    feats.append(atom.GetFormalCharge())
    # Aromaticity
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    # Ring membership
    feats.append(1.0 if atom.IsInRing() else 0.0)
    
    return np.array(feats, dtype=np.float32)

def bond_features(bond):
    """
    Generate bond features:
    - Bond type (one-hot: single, double, triple, aromatic)
    - Is conjugated (0 or 1)
    - Is in ring (0 or 1)
    """
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    feats = []
    feats.extend(one_of_k_encoding(bond.GetBondType(), bond_types))
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return np.array(feats, dtype=np.float32)

def molecule_to_graph(smiles):
    """
    Converts SMILES string into molecular graph tensors:
    - x: node (atom) features [N, num_atom_features]
    - edge_index: [2, 2*E] (undirected bonds)
    - edge_attr: edge (bond) features [2*E, num_bond_features]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    # Generate 3D Conformation
    mol_3d = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception:
        # Fallback if 3D conformer generation fails
        pass
    mol_3d = Chem.RemoveHs(mol_3d)
    
    # Get 3D coordinates
    conformer = mol_3d.GetConformer() if mol_3d.GetNumConformers() > 0 else None
    coords = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if conformer is not None:
            pos = conformer.GetAtomPosition(idx)
            coords.append([pos.x, pos.y, pos.z])
        else:
            coords.append([0.0, 0.0, 0.0])
            
    # Atom features
    atoms = mol.GetAtoms()
    x = [atom_features(atom) for atom in atoms]
    x = np.array(x, dtype=np.float32)
    
    # Bond features
    bonds = mol.GetBonds()
    edge_index = []
    edge_attr = []
    
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        f = bond_features(bond)
        
        # Undirected graph
        edge_index.append([i, j])
        edge_attr.append(f)
        edge_index.append([j, i])
        edge_attr.append(f)
        
    if len(edge_index) == 0:
        # Single atom molecules (e.g. Neon, Sodium)
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 6), dtype=np.float32)
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_attr = np.array(edge_attr, dtype=np.float32)
        
    return {
        'x': torch.tensor(x, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
        'coords': np.array(coords, dtype=np.float32),
        'atoms': [atom.GetSymbol() for atom in atoms]
    }

# ==========================================
# 2. Geometric / Message Passing GNN
# ==========================================

class MessagePassingLayer(nn.Module):
    """
    Custom GNN Message Passing Layer:
    m_i = sum_{j in N(i)} ReLU(Linear([h_i, h_j, e_ij]))
    h_i' = ReLU(Linear([h_i, m_i]))
    """
    def __init__(self, in_channels, edge_channels, out_channels):
        super(MessagePassingLayer, self).__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            # Self loop message if no bonds
            return F.relu(x)
            
        row, col = edge_index[0], edge_index[1]
        
        # Gather sender and receiver node states
        x_i = x[row]  # Target node
        x_j = x[col]  # Source node
        
        # Compute message
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        messages = self.message_mlp(msg_input)
        
        # Aggregate messages (Sum pooling)
        aggregated = torch.zeros((num_nodes, messages.size(1)), device=x.device)
        aggregated.index_add_(0, row, messages)
        
        # Update node state
        update_input = torch.cat([x, aggregated], dim=-1)
        x_new = self.update_mlp(update_input)
        
        return x_new

class MoleculeEncoder(nn.Module):
    """
    Core GDL GNN backbone. Represents a molecule as a graph and outputs
    a 128-dimensional embedding of the global molecular topology.
    """
    def __init__(self, num_atom_feats=31, num_bond_feats=6, hidden_dim=64):
        super(MoleculeEncoder, self).__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(num_atom_feats, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = MessagePassingLayer(hidden_dim, num_bond_feats, hidden_dim)
        self.conv2 = MessagePassingLayer(hidden_dim, num_bond_feats, hidden_dim)
        self.conv3 = MessagePassingLayer(hidden_dim, num_bond_feats, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # 1. Encode atoms
        h = self.atom_encoder(x)
        
        # 2. Message passing (3-bond context)
        h = h + F.relu(self.conv1(h, edge_index, edge_attr))
        h = h + F.relu(self.conv2(h, edge_index, edge_attr))
        h = h + F.relu(self.conv3(h, edge_index, edge_attr))
        
        # 3. Global Pooling (Sum + Max pooling) to form a unified fingerprint
        h_sum = torch.sum(h, dim=0, keepdim=True)
        h_max = torch.max(h, dim=0, keepdim=True)[0]
        h_mol = torch.cat([h_sum, h_max], dim=-1) # [1, hidden_dim * 2]
        return h_mol

class PredictionHead(nn.Module):
    """
    Task-specific classifier/regressor trained on the GDL molecular embedding.
    Allows modular plug-and-play adding of endpoints on demand.
    """
    def __init__(self, in_dim=128, hidden_dim=64):
        super(PredictionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_mol):
        return self.net(h_mol)

class MoleculeGDLModel(nn.Module):
    """
    Toxicity Predictor GNN wrapping the MoleculeEncoder and dynamic Task Heads.
    Maintains backward compatibility with original multi-task weight loading.
    """
    def __init__(self, num_atom_feats=31, num_bond_feats=6, hidden_dim=64):
        super(MoleculeGDLModel, self).__init__()
        # 1. Modular GDL backbone
        self.encoder = MoleculeEncoder(num_atom_feats, num_bond_feats, hidden_dim)
        
        # 2. Modular prediction heads (stored in a ModuleDict for dynamic indexing)
        self.heads = nn.ModuleDict({
            'herg': PredictionHead(in_dim=hidden_dim * 2, hidden_dim=hidden_dim),
            'ames': PredictionHead(in_dim=hidden_dim * 2, hidden_dim=hidden_dim),
            'dili': PredictionHead(in_dim=hidden_dim * 2, hidden_dim=hidden_dim),
            'ld50': PredictionHead(in_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        })

    def forward(self, x, edge_index, edge_attr):
        h_mol = self.encoder(x, edge_index, edge_attr)
        
        # Evaluate all registered heads in a deterministic order
        # We start with the 4 base endpoints to maintain strict index compatibility
        ordered_keys = ['herg', 'ames', 'dili', 'ld50']
        for k in self.heads.keys():
            if k not in ordered_keys:
                ordered_keys.append(k)
        
        logits = [self.heads[k](h_mol) for k in ordered_keys]
        return torch.cat(logits, dim=-1)

    def add_endpoint(self, endpoint_name, in_dim=128, hidden_dim=64):
        """
        Dynamically registers a new endpoint classification head on demand!
        """
        self.heads[endpoint_name] = PredictionHead(in_dim=in_dim, hidden_dim=hidden_dim)


# ==========================================
# 3. Integrated Gradients Explainability
# ==========================================

def compute_integrated_gradients(model, x, edge_index, edge_attr, target_class_idx, steps=30):
    """
    Computes Integrated Gradients attribution for each atom in the molecule graph.
    - x: node features [N, d]
    - target_class_idx: 0 (hERG), 1 (Ames), 2 (DILI), 3 (LD50)
    """
    model.eval()
    
    # Baseline: Zero features (neutral reference)
    x_baseline = torch.zeros_like(x)
    
    # Check if graph has edges
    has_edges = edge_index.numel() > 0
    
    # We will accumulate gradients of the prediction along the path from baseline to input
    grads = torch.zeros_like(x)
    
    for i in range(steps + 1):
        alpha = float(i) / steps
        x_step = x_baseline + alpha * (x - x_baseline)
        x_step = x_step.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = model(x_step, edge_index, edge_attr)
        probs = torch.sigmoid(logits)
        target_prob = probs[0, target_class_idx]
        
        # Backward pass
        model.zero_grad()
        target_prob.backward()
        
        # Accumulate gradients
        grads += x_step.grad.data
        
    # Average the gradients and multiply by input difference (x - x_baseline)
    avg_grads = grads / (steps + 1)
    attributions = (x - x_baseline) * avg_grads
    
    # Sum over feature dimension to get atom-level attribution scores
    atom_attributions = torch.sum(attributions, dim=-1).cpu().numpy()
    
    # Normalize scores to [-1, 1] range for visual heatmap
    max_val = np.max(np.abs(atom_attributions))
    if max_val > 0:
        atom_attributions = atom_attributions / max_val
        
    return atom_attributions
