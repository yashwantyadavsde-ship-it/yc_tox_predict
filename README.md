# 🔮 ToxGDL — Geometric Deep Learning Toxicity Predictor

> **Predict multi-endpoint molecular toxicity using a custom Graph Neural Network with integrated explainability.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)

---

## ✨ Overview

**ToxGDL** is an AI-powered drug safety analysis platform that predicts four critical toxicological endpoints for any molecule (input as a SMILES string) using a from-scratch **Geometric Deep Learning (GDL)** Graph Neural Network (GNN).

It also provides:
- 🧬 **Interactive 3D conformer visualization** with per-atom toxicophore heatmaps (via Integrated Gradients)
- 🤖 **Chemist Copilot** — auto-generated bioisosteric redesign suggestions to reduce toxicity
- 📊 **Molecular property panel** — MW, LogP, TPSA, H-bond donors/acceptors

---

## 🛡️ Toxicity Endpoints Predicted

| Endpoint | Description |
|---|---|
| **hERG Cardiotoxicity** | Off-target potassium channel blockade risk (QT prolongation) |
| **Ames Mutagenicity** | Genotoxic / DNA-damaging potential |
| **DILI Hepatotoxicity** | Drug-Induced Liver Injury risk |
| **Acute Oral Toxicity (LD50)** | High acute toxicity classification (LD50 < 50 mg/kg) |

---

## 🏗️ Architecture

```
SMILES Input
    │
    ▼
Molecular Graph Construction (RDKit)
  - Atom features: element, hybridization, degree, charge, aromaticity
  - Bond features: type, conjugation, ring membership
    │
    ▼
GDL Message Passing Network (3 layers)
  - Custom MessagePassingLayer (sum aggregation)
  - Skip connections (residual)
  - Global sum + max pooling
    │
    ▼
4x Independent Classification Heads (hERG / Ames / DILI / LD50)
    │
    ▼
Integrated Gradients → Per-Atom Attribution Scores
    │
    ▼
Chemist Copilot Advice + 3D Heatmap Visualization
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yashwantyadavsde-ship-it/yc_tox_predict.git
cd yc_tox_predict

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

> ⚡ On first run, if no pre-trained weights are found, the model will automatically train itself on the built-in curated toxicology reference dataset (~150 epochs, takes ~30 seconds).

---

## 📁 Project Structure

```
yc_tox_predict/
├── app.py                    # Flask application & prediction API
├── requirements.txt          # Python dependencies
├── model/
│   ├── gnn.py                # GNN architecture, featurization & Integrated Gradients
│   ├── train.py              # Training script & toxicology reference dataset
│   └── weights/              # Saved model weights (auto-generated, git-ignored)
│       └── gdl_toxicity_model.pt
├── templates/
│   └── index.html            # Jinja2 HTML template
└── static/
    ├── css/style.css         # Glassmorphism dark-mode UI styles
    └── js/main.js            # Frontend prediction logic & 3D viewer
```

---

## 🧪 Example Molecules

| Molecule | SMILES | Expected Signal |
|---|---|---|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | All safe ✅ |
| Haloperidol | `O=C(c1ccc(F)cc1)CCCN2CCC(n3c(=O)[nH]c4ccccc43)CC2` | hERG+ ⚠️ |
| Nitrobenzene | `O=N(=O)c1ccccc1` | Ames+ ⚠️ |
| Acetaminophen | `CC(=O)Nc1ccc(O)cc1` | DILI+ ⚠️ |
| Nicotine | `CN1CCCC1c2cccnc2` | LD50+ ⚠️ |

---

## ⚙️ Model Details

- **Framework**: PyTorch (custom GNN, no external GNN library required)
- **Training data**: 25 curated reference molecules with high-quality multi-endpoint toxicology labels
- **Loss**: Binary Cross-Entropy with Logits (multi-label)
- **Optimizer**: Adam (lr=0.005, weight_decay=1e-4)
- **Explainability**: Integrated Gradients (30 interpolation steps, zero-baseline)

> **Note**: This model is trained on a small reference set for demonstration purposes. For production use, training on larger datasets (e.g., Tox21, ClinTox, hERGdb) is strongly recommended.

---

## 📦 Dependencies

```
flask
rdkit
torch
numpy
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built with ❤️ using PyTorch, RDKit, and 3Dmol.js*
