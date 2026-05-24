# 🔮 ToxGDL: GDL Multi-Agent Discovery Lab & Custom Agent Builder

ToxGDL is a next-generation, GDL-powered **Multi-Agent Scientific Discovery Laboratory** designed to help senior drug discovery researchers identify, explain, and optimize away toxicological and ADMET safety liabilities in drug candidates.

By merging **explainable Geometric Deep Learning (GDL)** with **RDKit-based organic chemistry mutation rules**, **Isomorphic Labs-inspired 3D pocket docking**, and an **autonomous board of collaborating AI specialists**, ToxGDL turns passive toxicity calculators into an active, collaborative discovery partner.

---

## 🚀 Key Architectural Pillars

```
             ┌──────────────────────────────────────────────┐
             │         INPUT: Chemical SMILES String        │
             └──────────────────────┬───────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   1. GDL FOUNDATION ENGINE (PyTorch GNN Backbone)       │
       │   - 3-Layer Message Passing (Bonds & Atoms)             │
       │   - Integrated Gradients: 3D Toxicophore Heatmaps       │
       │   - Isomorphic 3D Target Docking (Kd & Binding Energy)   │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   2. EXPERT ORGANIC MUTATION (RDKit SMARTS Reactions)   │
       │   - Ames & hERG Bioisosteric Swaps (Amides, Ethers)     │
       │   - Targeted C-H -> C-F Fluorine Metabolic Shielding    │
       │   - Standalone Topological Synthetic Feasibility (SA)    │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   3. AUTONOMOUS BOARDROOM (Multi-Agent Panel)           │
       │   - Dr. GDL (GNN Physicist)                             │
       │   - Dr. Synthesis (Medicinal Chemist)                   │
       │   - Dr. ADME (Pharmacokinetics Expert)                  │
       │   - Dr. Safety (Lead Toxicologist)                      │
       │   - CUSTOM DEPLOYED AGENTS (FDA Audit, Prof. Soluble)   │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
            ┌──────────────────────────────────────────────┐
            │   OUTPUT: Interactive MPO Comparison Cards   │
            │   & Collaborative Dialogue Consultation Log  │
            └──────────────────────────────────────────────┘
```

### 1. The GDL Foundation Engine (`model/gnn.py`)
At the core of ToxGDL is a **Geometric Deep Learning (GDL)** network built in PyTorch. Unlike standard models that process chemicals as text strings, GDL treats molecules in their natural, non-Euclidean representation—a **stereochemical graph** of nodes (atoms) and edges (bonds).
* **Backbone Encoder (`MoleculeEncoder`):** A 3-layer Message Passing Neural Network (MPNN) that propagates chemical structural features up to 3 bonds away, pooling them into a 128-dimensional global molecular topology embedding.
* **Plug-and-Play heads:** Decoupled classification heads (`PredictionHead`) to evaluate endpoints on demand (hERG cardiotoxicity, Ames mutagenicity, DILI hepatotoxicity, and acute LD50 oral toxicity).
* **Explainability (Integrated Gradients):** Runs path-integral calculus on the GNN's outputs with respect to graph nodes. This mathematically calculates exactly which individual atoms drive toxicity, lighting them up in **bright red (toxicophores)** inside a **3Dmol.js conformer viewer**.

### 2. Isomorphic Labs-Inspired Target Docking (`model/copilot.py`)
Rather than predicting toxicity in isolation, ToxGDL incorporates structure-based target interaction. Select a therapeutic pocket (e.g. *EGFR oncology targets*, *Kinase cavities*, *CYP3A4 metabolic pockets*) in the sidebar, and the system simulates **3D Pocket Docking**:
* Computes shape complementarity, hydrophobic surface matching (LogP curves), polar networks (TPSA, hydrogen bonding), and rotatable bond entropy loss.
* Estimates **Gibbs Free Energy of Binding ($\Delta G$ in kcal/mol)** and **dissociation constants ($K_d$ in nanomolar)** to calculate the crucial **Selectivity Index** (Target-to-Tox ratio).

### 3. Collaborative Multi-Agent Advisory Board (`model/agents.py`)
ToxGDL features an autonomous panel of specialized AI researchers that debate trade-offs and collaborate to form a consensus redesign report:
* **Dr. GDL (GNN Structural Physicist):** Maps conformer coordinates and toxicophore attribution hotspots.
* **Dr. Synthesis (Medicinal Chemist):** Suggests realistic organic changes, bioisostere transforms, and synthetic feasibility (SA Score).
* **Dr. ADME (Pharmacokinetics Expert):** Evaluates LogP, MW, TPSA, and membrane permeability trade-offs.
* **Dr. Safety (Lead Toxicologist):** Reviews GDL scores, aligns with FDA boundaries, and signs off on the final consensus.
* **Anthropic-Style Structured Reasoning:** Each agent runs a hidden **System 2 "thought" process** (doing deep mathematical analysis) before outputting its public **dialogue** to the boardroom chat thread.

### 4. Custom Agent Laboratory (🧬 Deploy Portal)
Senior researchers can **scale the lab by spinning up custom AI agents on the fly**:
* Provide an Agent Name, emoji avatar, scientific role/specialty, and a custom persona prompt (defining their analytical reasoning bias).
* **Knowledge Resources Field:** Paste custom links, research papers, or database guidelines. The agent will dynamically incorporate specific quotes and scientific principles from those resources to justify its claims during the boardroom debate!

---

## 💻 Installation & Running Locally

### Prerequisites
* **Python 3.8+**
* **RDKit** (for chemical graph manipulations)
* **PyTorch** (for GNN tensor calculations)
* **Flask** (for web controller routing)

### Setup & Run
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yashwantyadavsde-ship-it/yc_tox_predict.git
   cd yc_tox_predict
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Calibrate the GNN Models (Bootstrap Weights):**
   Run the training script to train the backbone encoder and classification task heads on the reference toxicology datasets:
   ```bash
   python model/train.py
   ```
   *The calibrated weights will be saved to `model/weights/gdl_toxicity_model.pt`.*

5. **Start the Discovery Lab Server:**
   ```bash
   python app.py
   ```

6. **Open in Your Browser:**
   Go to `http://127.0.0.1:5000` to launch the high-tech, glassmorphic Multi-Agent console.

---

## ⚡ The Iterative Design Workflow: What to Expect

1. **Analyze a Lead:** Input a 2D/3D SMILES string (e.g. *Haloperidol* presets) and select your therapeutic target in the sidebar. Click **⚡ Run Geometric Analysis**.
2. **Observe the Boardroom:** Dr. GDL, Dr. Synthesis, Dr. ADME, and Dr. Safety immediately hold a collaborative debate in the **AI Research Board Room** tab. Toggle **View System 2 Thought** on any message to audit their hidden reasoning.
3. **Review the MPO Cards:** In the right panel, inspect the synthesizable bioisostere cards, including their SA Scores, target pocket affinity, and side-by-side comparison tables with color-coded positive/negative changes.
4. **Deploy Custom Agents:** Switch to the **🧬 Custom Agent Laboratory** tab. Deploy a custom agent named *Dr. FDA*, avatar `🛡️`, who flags Lipinski Rule of 5 violations and references your pasted guidelines. Deploy him, re-run your compound, and watch him instantly join the chat thread!
5. **Close the Loop:** Click **⚡ Load Analog** on a proposed safe compound. It instantly loads back into the main analyzer, generates its new 3D shape and safety stats, and lets you dynamically refine your molecule, creating an addictive, high-velocity drug-design loop!

---

## 🛡️ License & Contributions
ToxGDL is open for contributions and academic research integrations. Licensed under the MIT License.
