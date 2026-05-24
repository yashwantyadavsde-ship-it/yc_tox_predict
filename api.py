import os
import sys
import torch

# Ensure model directory is on path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.gnn import MoleculeGDLModel, molecule_to_graph

# Try to import FastAPI/Pydantic for the API server
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ToxGDL: Toxicity Predictor API",
        description="Predicts off-target toxicity using Geometric Deep Learning",
        version="1.0.0"
    )

    # Load Model weights
    MODEL_WEIGHTS_PATH = os.path.join("model", "weights", "gdl_toxicity_model.pt")
    model = MoleculeGDLModel(num_atom_feats=31, num_bond_feats=6, hidden_dim=64)

    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            state_dict = torch.load(MODEL_WEIGHTS_PATH, weights_only=True)
            # Map legacy keys if needed
            new_state_dict = {}
            for key, val in state_dict.items():
                if key.startswith("atom_encoder") or key.startswith("conv1") or key.startswith("conv2") or key.startswith("conv3"):
                    new_key = "encoder." + key
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
            model.eval()
            print("API GNN Model loaded successfully with decoupled compatibility.")
        except Exception as e:
            print(f"Failed to load GNN weights: {e}. Running with randomized weights.")
    else:
        print("WARNING: Model weights not found. Running with randomized weights. Please run model/train.py first.")

    class PredictionRequest(BaseModel):
        smiles: str

    class PredictionResponse(BaseModel):
        smiles: str
        herg_score: float
        ames_score: float
        dili_score: float
        ld50_score: float
        flags: list[str]

    @app.post("/predict_toxicity", response_model=PredictionResponse)
    def predict_toxicity(request: PredictionRequest):
        smiles = request.smiles.strip()
        graph = molecule_to_graph(smiles)
        
        if graph is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string or empty molecule.")
            
        x, edge_index, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr']
        
        with torch.no_grad():
            logits = model(x, edge_index, edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        flags = []
        if probs[0] > 0.5: flags.append("HIGH_HERG_CARDIOTOX_RISK")
        if probs[1] > 0.5: flags.append("HIGH_AMES_MUTAGEN_RISK")
        if probs[2] > 0.5: flags.append("HIGH_DILI_HEPATOTOX_RISK")
        if probs[3] > 0.5: flags.append("HIGH_ACUTE_LD50_RISK")
        
        if not flags:
            flags.append("SAFE_PROFILE")
            
        return PredictionResponse(
            smiles=smiles,
            herg_score=round(float(probs[0]), 4),
            ames_score=round(float(probs[1]), 4),
            dili_score=round(float(probs[2]), 4),
            ld50_score=round(float(probs[3]), 4),
            flags=flags
        )

    @app.get("/")
    def read_root():
        return {"message": "Welcome to the ToxGDL API. Use POST /predict_toxicity"}

else:
    # Fallback explanation if FastAPI is executed without installing dependencies
    def main():
        print("ToxGDL API Port Controller")
        print("--------------------------")
        print("To run the FastAPI server, please install the optional dependencies first:")
        print("  pip install fastapi uvicorn pydantic")
        print("\nNote: The primary entry point for this application is Flask ('python app.py').")
        
    if __name__ == '__main__':
        main()
