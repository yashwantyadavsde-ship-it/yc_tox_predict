import os
import sys
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np

# Ensure root directory is on the path so we can import model
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.gnn import MoleculeGDLModel, molecule_to_graph, compute_integrated_gradients
from model.train import train_and_save

app = Flask(__name__)

# Bootstrapping: Ensure model is trained and weights exist
MODEL_WEIGHTS_PATH = os.path.join("model", "weights", "gdl_toxicity_model.pt")
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print("Model weights not found. Bootstrapping and training the GNN now...")
            train_and_save()
        
        # Load weights
        model = MoleculeGDLModel(num_atom_feats=31, num_bond_feats=6, hidden_dim=64)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
        model.eval()
        print("GNN Model successfully loaded.")
    return model

# Dictionary mapping endpoint indexes to names
ENDPOINTS = ["hERG Cardiotoxicity", "Ames Mutagenicity", "DILI Hepatotoxicity", "Acute Oral Toxicity (LD50)"]

def generate_chemist_copilot_advice(smiles, mol, attributions, predictions):
    """
    Analyzes toxicophores and gives scientific suggestions to modify the molecule.
    """
    suggestions = []
    
    # 1. Identify toxic endpoints (threshold > 0.5)
    toxic_endpoints = [ENDPOINTS[i] for i, p in enumerate(predictions) if p > 0.5]
    
    if not toxic_endpoints:
        return ["This molecule shows a very clean safety profile! No major toxicophores detected."]

    # 2. Find the top toxicophore atom
    # Average attribution across toxic endpoints
    avg_attrs = np.zeros(len(mol.GetAtoms()))
    active_endpoints_count = 0
    for idx, p in enumerate(predictions):
        if p > 0.5:
            avg_attrs += attributions[idx]
            active_endpoints_count += 1
            
    if active_endpoints_count > 0:
        avg_attrs = avg_attrs / active_endpoints_count
        
    top_atom_idx = int(np.argmax(avg_attrs))
    top_atom = mol.GetAtomWithIdx(top_atom_idx)
    symbol = top_atom.GetSymbol()
    
    # 3. Analyze specific functional groups and off-target risks
    # Ames Mutagenicity suggestions
    if predictions[1] > 0.5: # Ames Mutagen
        # Check for nitro groups
        has_nitro = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and any(n.GetSymbol() == 'O' for n in atom.GetNeighbors()):
                has_nitro = True
                break
        if has_nitro:
            suggestions.append("⚠️ **Ames Mutagenicity Warning**: Detected a highly mutagenic **Nitro group (-NO2)**. Consider replacing it with a Bioisostere like an **Amide (-NHCO-R)** or a **Sulfonamide (-SO2NH-R)** to eliminate mutagenicity while retaining potential hydrogen bond properties.")
        else:
            suggestions.append("⚠️ **Ames Mutagenicity Warning**: Model identified mutagenic potential, likely driven by aromatic nitrogen/amine motifs. Consider introducing a **Fluorine atom** or **Steric hindrance** around the primary amine to decrease reactive metabolite formation.")

    # hERG Cardiotoxicity suggestions
    if predictions[0] > 0.5: # hERG Cardiotoxicity
        # Bulky basic nitrogens?
        has_basic_nitrogen = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and not atom.GetIsAromatic():
                has_basic_nitrogen = True
                break
        if has_basic_nitrogen:
            suggestions.append("⚠️ **hERG Cardiotoxicity Warning**: High off-target risk of QT interval prolongation. Basic tertiary/secondary amines often bind strongly to the hERG potassium channel. Try: \n   - Lowering the pKa of the amine by adding electron-withdrawing groups (e.g. introducing Fluorines near it).\n   - Replacing the basic amine with an amide or ether oxygen to reduce ionic interactions.")
        else:
            suggestions.append("⚠️ **hERG Cardiotoxicity Warning**: High hERG binding risk. Try reducing overall **Lipophilicity (logP)** by adding polar groups (e.g., hydroxyls or carboxylates) or breaking aromatic symmetry.")

    # DILI Hepatotoxicity suggestions
    if predictions[2] > 0.5: # DILI
        suggestions.append("⚠️ **Hepatotoxicity Warning**: Detected liver toxicity risk. Consider shielding active metabolic sites by adding **Fluorine** blockers or reducing lipophilicity to lower high clearance rates by cytochrome P450 enzymes.")

    if not suggestions:
        suggestions.append("💡 Consider adjusting the highly polarized regions (red atoms in 3D viewer) to decrease non-specific cell binding.")
        
    return suggestions

def get_sdf_string(mol):
    """
    Returns the 3D structure Mol Block representation.
    """
    try:
        return Chem.MolToMolBlock(mol)
    except Exception:
        return ""

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'smiles' not in data:
        return jsonify({'error': 'No SMILES string provided.'}), 400
        
    smiles = data['smiles'].strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': 'Invalid SMILES string. Please check the chemical structure.'}), 400
        
    # Generate 3D Conformation
    mol_3d = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception:
        pass
    mol_3d = Chem.RemoveHs(mol_3d)
    
    # Run graph extraction
    graph = molecule_to_graph(smiles)
    if graph is None:
        return jsonify({'error': 'Failed to process molecule graph representation.'}), 400
        
    # Retrieve GNN model
    net = get_model()
    
    # Run prediction
    x, edge_index, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr']
    with torch.no_grad():
        logits = net(x, edge_index, edge_attr)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    # Run Integrated Gradients for each class
    attributions = []
    for c_idx in range(len(ENDPOINTS)):
        attr = compute_integrated_gradients(net, x, edge_index, edge_attr, target_class_idx=c_idx)
        attributions.append(attr.tolist())
        
    # Generate Copilot Advice
    advice = generate_chemist_copilot_advice(smiles, mol, attributions, probs)
    
    # Get 3D SDF coordinates
    sdf_data = get_sdf_string(mol_3d)
    
    # Compute molecular weight
    mw = round(Chem.rdMolDescriptors.CalcExactMolWt(mol), 2)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    logp = round(Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0], 2)
    hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
    hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
    tpsa = round(Chem.rdMolDescriptors.CalcTPSA(mol), 2)
    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    return jsonify({
        'smiles': smiles,
        'formula': formula,
        'mw': mw,
        'logp': logp,
        'hbd': hbd,
        'hba': hba,
        'tpsa': tpsa,
        'rot_bonds': rot_bonds,
        'predictions': {
            'herg': float(probs[0]),
            'ames': float(probs[1]),
            'dili': float(probs[2]),
            'ld50': float(probs[3])
        },
        'attributions': {
            'herg': attributions[0],
            'ames': attributions[1],
            'dili': attributions[2],
            'ld50': attributions[3]
        },
        'sdf': sdf_data,
        'atoms': graph['atoms'],
        'advice': advice
    })

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)
