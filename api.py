from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from graph_builder import smiles_to_graph
from train_gnn import ToxicityGNN

app = FastAPI(
    title="Digital Firewall: Toxicity Predictor",
    description="Predicts off-target toxicity using Geometric Deep Learning",
    version="1.0.0"
)

# Load Model
model = ToxicityGNN(hidden_channels=64)
try:
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("WARNING: model.pth not found. The model will use random weights.")

class PredictionRequest(BaseModel):
    smiles: str

class PredictionResponse(BaseModel):
    smiles: str
    risk_score: float
    flags: list[str]

@app.post("/predict_toxicity", response_model=PredictionResponse)
def predict_toxicity(request: PredictionRequest):
    smiles = request.smiles
    
    # Convert SMILES to Graph (Pass dummy toxicity label 0)
    graph = smiles_to_graph(smiles, y_val=0)
    
    if graph is None or graph.num_nodes == 0:
        raise HTTPException(status_code=400, detail="Invalid SMILES string or empty molecule.")
        
    # We need to add a batch vector since the model expects batch indices
    # For a single graph, the batch vector is just zeros
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch)
        # Apply sigmoid to get probability
        prob = torch.sigmoid(out).item()
        
    # Create flags based on risk
    flags = []
    if prob > 0.8:
        flags.append("CRITICAL_TOXICITY_RISK")
    elif prob > 0.5:
        flags.append("HIGH_TOXICITY_RISK")
    else:
        flags.append("SAFE")
        
    return PredictionResponse(
        smiles=smiles,
        risk_score=round(prob, 4),
        flags=flags
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Digital Firewall API. Use POST /predict_toxicity"}
