import os
import sys
import requests
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np

# Ensure root directory is on the path so we can import model
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.gnn import MoleculeGDLModel, molecule_to_graph, compute_integrated_gradients
from model.train import train_and_save
from model.copilot import mutate_molecule, estimate_target_affinity
from model.agents import global_registry, run_consultation

LOCAL_NAME_TO_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "haloperidol": "O=C(c1ccc(F)cc1)CCCN2CCC(n3c(=O)[nH]c4ccccc43)CC2",
    "nicotine": "CN1CCCC1c2cccnc2",
    "caffeine": "CN1C(=O)N(C)c2ncn(C)c2C1=O",
    "water": "O",
    "ethanol": "CCO",
    "methanol": "CO",
    "glucose": "OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O",
    "cocaine": "CN1C2CCC1C(C(=O)OC)C(OC(=O)c3ccccc3)C2"
}

def resolve_name_to_smiles(name):
    """
    Resolves common drug names / chemical names to SMILES strings.
    Utilizes a local fast-lookup dictionary and falls back dynamically 
    to the official NIH PubChem PUG REST API.
    """
    name_clean = name.strip().lower()
    
    # 1. Local fast dictionary lookup
    if name_clean in LOCAL_NAME_TO_SMILES:
        return LOCAL_NAME_TO_SMILES[name_clean]
        
    # 2. Remote NIH PubChem PUG REST API query
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_clean}/property/CanonicalSMILES/JSON"
        res = requests.get(url, timeout=2.0)
        if res.status_code == 200:
            data = res.json()
            smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return smiles
    except Exception as e:
        print(f"PubChem Name resolution failed for '{name}': {e}")
        
    return None

app = Flask(__name__)

# Bootstrapping: Ensure model is trained and weights exist
MODEL_WEIGHTS_PATH = os.path.join("model", "weights", "gdl_toxicity_model.pt")
model = None

def load_state_dict_compatible(model, state_dict):
    """
    Translates weights from legacy multi-task models into the new
    decoupled backbone encoder + prediction heads format dynamically.
    """
    new_state_dict = {}
    for key, val in state_dict.items():
        # 1. Map encoder/backbone keys
        if key.startswith("atom_encoder") or key.startswith("conv1") or key.startswith("conv2") or key.startswith("conv3"):
            new_key = "encoder." + key
        # 2. Map classification heads
        elif key.startswith("fc_herg"):
            new_key = key.replace("fc_herg", "heads.herg.net")
        elif key.startswith("fc_ames"):
            new_key = key.replace("fc_ames", "heads.ames.net")
        elif key.startswith("fc_dili"):
            new_key = key.replace("fc_dili", "heads.dili.net")
        elif key.startswith("fc_ld50"):
            new_key = key.replace("fc_ld50", "heads.ld50.net")
        else:
            new_key = key
        new_state_dict[new_key] = val
    model.load_state_dict(new_state_dict)

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print("Model weights not found. Bootstrapping and training the GNN now...")
            train_and_save()
        
        # Load weights with dynamic compatibility mapping
        model = MoleculeGDLModel(num_atom_feats=31, num_bond_feats=6, hidden_dim=64)
        state_dict = torch.load(MODEL_WEIGHTS_PATH, weights_only=True)
        load_state_dict_compatible(model, state_dict)
        model.eval()
        print("GNN Model successfully loaded with decoupled compatibility.")
    return model

def score_molecule_for_copilot(smiles):
    """
    Evaluation subroutine passed to the copilot engine to compute predictions
    of generated bioisosteres in real-time.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    graph = molecule_to_graph(smiles)
    if graph is None:
        return None
    net = get_model()
    x, edge_index, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr']
    with torch.no_grad():
        logits = net(x, edge_index, edge_attr)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {
        "mol": mol,
        "predictions": probs
    }

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
        return jsonify({'error': 'No SMILES or chemical name provided.'}), 400
        
    smiles = data['smiles'].strip()
    if not smiles:
        return jsonify({'error': 'Chemical input cannot be empty.'}), 400
        
    # Check if the input is a common chemical/drug name instead of a SMILES string
    mol = Chem.MolFromSmiles(smiles)
    resolved_name = None
    
    if mol is None:
        # It's not a valid SMILES. Let's resolve it dynamically using PubChem or local dict
        resolved_smiles = resolve_name_to_smiles(smiles)
        if resolved_smiles is not None:
            resolved_name = smiles # Keep track of original chemical name
            smiles = resolved_smiles
            mol = Chem.MolFromSmiles(smiles)
            
    is_general_query = False
    if mol is None:
        # Check if the text looks like a general scientific query rather than a malformed chemical name
        if len(smiles.split()) > 1 or "?" in smiles or len(smiles) > 25:
            is_general_query = True
            
    if is_general_query:
        target_protein = data.get('target', 'General').strip()
        consultation_log = run_consultation(smiles, parent_data=None, analogs=[], target_protein=target_protein)
        return jsonify({
            'is_general_query': True,
            'query': smiles,
            'consultation_log': consultation_log
        })
            
    if mol is None:
        return jsonify({'error': f"Failed to resolve chemical representation. '{data['smiles']}' is not a valid SMILES string, was not recognized as a common drug name, and does not match the pattern of a general scientific query. Please verify spelling."}), 400
        
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
    
    # Compute molecular weight and descriptors
    mw = round(Chem.rdMolDescriptors.CalcExactMolWt(mol), 2)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    logp = round(Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0], 2)
    hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
    hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
    tpsa = round(Chem.rdMolDescriptors.CalcTPSA(mol), 2)
    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    # Extract target protein from POST body
    target_protein = data.get('target', 'General').strip()
    
    # Calculate 3D Target Pocket Affinity docking stats for parent
    parent_affinity = estimate_target_affinity(mol, target_protein)
    
    # Calculate top toxicophore atom idx for targeted mutations
    avg_attrs = np.zeros(len(mol.GetAtoms()))
    active_endpoints_count = 0
    for idx, p in enumerate(probs):
        if p > 0.5:
            avg_attrs += attributions[idx]
            active_endpoints_count += 1
            
    if active_endpoints_count > 0:
        avg_attrs = avg_attrs / active_endpoints_count
    top_atom_idx = int(np.argmax(avg_attrs)) if len(mol.GetAtoms()) > 0 else 0
    
    # Run RDKit organic chemistry mutation and scoring engine (passing target protein)
    analogs = mutate_molecule(smiles, probs, top_atom_idx, score_molecule_for_copilot, target_protein)
    
    # Execute the GDL-powered Multi-Agent Boardroom Consultation dialogue
    parent_payload = {
        "predictions": {
            "herg": float(probs[0]),
            "ames": float(probs[1]),
            "dili": float(probs[2]),
            "ld50": float(probs[3])
        },
        "logp": logp,
        "mw": mw,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "formula": formula
    }
    consultation_log = run_consultation(smiles, parent_payload, analogs, target_protein)
    
    # Get 3D SDF coordinates
    sdf_data = get_sdf_string(mol_3d)
    
    return jsonify({
        'smiles': smiles,
        'formula': formula,
        'mw': mw,
        'logp': logp,
        'hbd': hbd,
        'hba': hba,
        'tpsa': tpsa,
        'rot_bonds': rot_bonds,
        'affinity': parent_affinity,
        'resolved_name': resolved_name,
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
        'advice': advice,
        'analogs': analogs,
        'consultation_log': consultation_log
    })

@app.route('/create_agent', methods=['POST'])
def create_agent():
    """
    POST API to dynamically register custom scientific agents inside
    the global Agent Registry in real-time.
    """
    data = request.get_json()
    if not data or 'name' not in data or 'persona' not in data:
        return jsonify({'error': 'Name and Persona prompt are required fields.'}), 400
        
    name = data['name'].strip()
    role = data.get('role', 'Specialized Scientist').strip()
    avatar = data.get('avatar', '🧬').strip()
    persona = data['persona'].strip()
    resources = data.get('resources', '').strip()
    
    success = global_registry.register_agent(name, role, avatar, persona, resources)
    
    if success:
        return jsonify({'message': f"Agent '{name}' successfully deployed to active boardroom!"})
    else:
        return jsonify({'error': 'Failed to deploy agent. Name cannot be empty.'}), 400

@app.route('/get_agents', methods=['GET'])
def get_agents():
    """
    GET API to return list of all active AI agents.
    """
    return jsonify(global_registry.get_all_agents())

@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self' http://127.0.0.1:* http://localhost:* https://cdnjs.cloudflare.com; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob: *; "
        "connect-src 'self' http://127.0.0.1:* http://localhost:* https://pubchem.ncbi.nlm.nih.gov; "
        "frame-ancestors 'self' *;"
    )
    return response

if __name__ == '__main__':
    # Start the Flask app on a clean, fresh port
    app.run(host='127.0.0.1', port=5080, debug=True)
