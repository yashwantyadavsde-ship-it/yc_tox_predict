import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# =====================================================================
# 1. Standalone Topological Synthetic Accessibility (SA) Scorer
# =====================================================================

def calculate_sa_score(mol):
    """
    Computes a robust, standalone topological Synthetic Accessibility (SA) Score (1 to 10 scale).
    - 1.0 represents extremely easy to synthesize (like simple chains/alcohols).
    - 10.0 represents practically impossible/extremely complex molecules.
    Estimates based on:
    - Molecular weight (complexity scaling)
    - Number of rings (especially medium, large, or bridged rings)
    - Number of stereocenters / chiral atoms
    - Presence of uncommon/reactive atoms or complex fusion points
    """
    if mol is None:
        return 5.0
        
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    
    # 1. Base complexity from size (atom/bond counts)
    size_penalty = 1.0 + (num_atoms * 0.05) + (num_bonds * 0.05)
    
    # 2. Ring penalty (especially fused or macrocycles)
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()
    ring_penalty = 0.0
    if num_rings > 0:
        ring_penalty = num_rings * 0.3
        # Check for large or bridged rings
        for ring in ring_info.AtomRings():
            if len(ring) > 8:
                ring_penalty += 0.5  # Macrocycle penalty
            elif len(ring) < 5:
                ring_penalty += 0.2  # High-strain rings (3- or 4-membered)
                
    # 3. Chiral centers / Stereocenters penalty
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    stereo_penalty = len(chiral_centers) * 0.8
    
    # 4. Heteroatom complexity (halogenated, sulfur/phosphorus density)
    heteroatoms = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ['C', 'H', 'O', 'N']:
            heteroatoms += 1
    hetero_penalty = heteroatoms * 0.25
    
    # Combined raw score
    raw_score = size_penalty + ring_penalty + stereo_penalty + hetero_penalty
    
    # Normalize between 1.0 and 10.0
    normalized_score = min(10.0, max(1.0, round(raw_score, 2)))
    return normalized_score


# =====================================================================
# 2. Bioisosteric Chemical Reaction Rules
# =====================================================================

# Define standard chemical reactions using SMARTS transformation rules
MUTATION_RULES = {
    # 1. Ames Nitro Group replacements
    "nitro_to_acetamide": {
        "smarts": "[c,C:1]-[N+:2](=[O:3])[O-:4]>>[c,C:1]-[NH:2]-C(=O)-C",
        "description": "Replace mutagenic Nitro group with stable Acetamide bioisostere to eliminate genotoxic potential.",
        "type": "Ames"
    },
    "nitro_to_sulfonamide": {
        "smarts": "[c,C:1]-[N+:2](=[O:3])[O-:4]>>[c,C:1]-S(=O)(=O)-[NH2:2]",
        "description": "Replace Nitro group with Sulfonamide group to preserve polar hydrogen bonding while removing mutagenicity.",
        "type": "Ames"
    },
    
    # 2. hERG Basic Amine replacements (reducing cardiotoxicity)
    "sec_amine_to_amide": {
        "smarts": "[C:1]-[NH1:2]-[C:3]>>[C:1]-N(-[C:3])-C(=O)-C",
        "description": "Convert basic secondary amine to amide to dramatically lower pKa, eliminating ionic interactions with hERG channels.",
        "type": "hERG"
    },
    "tert_amine_to_amide": {
        "smarts": "[C:1]-[N;H0;D3:2](-[C:3])-[C:4]>>[C:1]-N(-[C:3])-C(=O)-C",
        "description": "Convert basic tertiary amine to non-basic amide, removing hERG channel binding affinity.",
        "type": "hERG"
    },
    "sec_amine_to_ether": {
        "smarts": "[C:1]-[NH1:2]-[C:3]>>[C:1]-[O:2]-[C:3]",
        "description": "Replace secondary amine with neutral ether link to remove charge and avoid cardiac QT prolongation.",
        "type": "hERG"
    },
    "tert_amine_to_ether": {
        "smarts": "[C:1]-[N;H0;D3:2](-[C:3])-[C:4]>>[C:1]-[O:2]-[C:3]",
        "description": "Replace tertiary amine with neutral ether to neutralize basic nitrogen hERG binding.",
        "type": "hERG"
    }
}


# =====================================================================
# 3. Core Mutation Engine
# =====================================================================

def mutate_molecule(smiles, original_predictions, top_atom_idx, evaluate_func):
    """
    Takes a parent compound, identifies toxic endpoints and top toxicophores,
    and applies RDKit chemical mutations to design safe, drug-like analogs.
    
    - smiles: Parent SMILES string.
    - original_predictions: List of original predictions [herg, ames, dili, ld50]
    - top_atom_idx: Index of highest attribution atom from Integrated Gradients.
    - evaluate_func: Callable that evaluates a SMILES string, returning predictions and RDKit molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    parent_smiles = Chem.MolToSmiles(mol, canonical=True)
    suggested_analogs = []
    seen_smiles = {parent_smiles}

    herg_toxic = original_predictions[0] > 0.5
    ames_toxic = original_predictions[1] > 0.5
    dili_toxic = original_predictions[2] > 0.5

    # -------------------------------------------------------------
    # Rule 1: Reaction-based bioisostere swaps (Ames or hERG)
    # -------------------------------------------------------------
    for rule_name, rule_data in MUTATION_RULES.items():
        # Check if the rule matches the predicted toxicity profile
        if rule_data["type"] == "Ames" and not ames_toxic:
            continue
        if rule_data["type"] == "hERG" and not herg_toxic:
            continue
            
        try:
            rxn = AllChem.ReactionFromSmarts(rule_data["smarts"])
            products = rxn.RunReactants((mol,))
            
            for prod_list in products:
                prod = prod_list[0]
                try:
                    Chem.SanitizeMol(prod)
                    prod_smiles = Chem.MolToSmiles(prod, canonical=True)
                    
                    if prod_smiles not in seen_smiles and "." not in prod_smiles:
                        seen_smiles.add(prod_smiles)
                        suggested_analogs.append({
                            "smiles": prod_smiles,
                            "change": rule_data["description"],
                            "source": "reaction"
                        })
                except Exception:
                    continue
        except Exception as e:
            print(f"Failed to run reaction rule {rule_name}: {e}")

    # -------------------------------------------------------------
    # Rule 2: Targeted Atom Mutation (Fluorine metabolic block on DILI/Tox atom)
    # -------------------------------------------------------------
    # If DILI is toxic, or generally if we have a top atom, we can introduce a Fluorine metabolic blocker
    if (dili_toxic or herg_toxic or ames_toxic) and top_atom_idx is not None and top_atom_idx < mol.GetNumAtoms():
        try:
            target_atom = mol.GetAtomWithIdx(int(top_atom_idx))
            
            # We can only place Fluorine blocker on a Carbon atom that possesses Hydrogens to be replaced
            if target_atom.GetSymbol() == 'C':
                # Create a mutable copy of the molecule
                rw_mol = Chem.RWMol(mol)
                
                # Create Fluorine atom
                f_atom = Chem.Atom(9)  # Fluorine atomic number is 9
                f_idx = rw_mol.AddAtom(f_atom)
                
                # Add single bond from the target carbon to Fluorine
                rw_mol.AddBond(int(top_atom_idx), f_idx, Chem.BondType.SINGLE)
                
                # Sanitize and compile
                prod = rw_mol.GetMol()
                Chem.SanitizeMol(prod)
                prod_smiles = Chem.MolToSmiles(prod, canonical=True)
                
                if prod_smiles not in seen_smiles and "." not in prod_smiles:
                    seen_smiles.add(prod_smiles)
                    suggested_analogs.append({
                        "smiles": prod_smiles,
                        "change": f"⚠️ **Metabolic Shielding**: Introduced a **Fluorine atom** on Carbon-#{top_atom_idx} to block enzyme oxidation and shield active metabolic toxicophores.",
                        "source": "targeted_mutation"
                    })
        except Exception as e:
            print(f"Failed to perform fluorine targeted mutation: {e}")

    # -------------------------------------------------------------
    # 4. Property Evaluation and Real-Time Re-Scoring
    # -------------------------------------------------------------
    scored_analogs = []
    
    # Cap at top 3 high-quality suggestions to keep response times fast and user interface clean
    for analog in suggested_analogs[:3]:
        analog_smiles = analog["smiles"]
        
        # Evaluate analog using the GNN server function
        eval_result = evaluate_func(analog_smiles)
        if eval_result is None:
            continue
            
        analog_mol = eval_result["mol"]
        analog_preds = eval_result["predictions"]
        
        # Calculate physical descriptors
        mw = round(Chem.rdMolDescriptors.CalcExactMolWt(analog_mol), 2)
        logp = round(Chem.rdMolDescriptors.CalcCrippenDescriptors(analog_mol)[0], 2)
        tpsa = round(Chem.rdMolDescriptors.CalcTPSA(analog_mol), 2)
        hbd = Chem.rdMolDescriptors.CalcNumHBD(analog_mol)
        hba = Chem.rdMolDescriptors.CalcNumHBA(analog_mol)
        sa_score = calculate_sa_score(analog_mol)
        
        scored_analogs.append({
            "smiles": analog_smiles,
            "change": analog["change"],
            "mw": mw,
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "sa_score": sa_score,
            "predictions": {
                "herg": float(analog_preds[0]),
                "ames": float(analog_preds[1]),
                "dili": float(analog_preds[2]),
                "ld50": float(analog_preds[3])
            }
        })
        
    return scored_analogs
