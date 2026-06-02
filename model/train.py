import os
import sys
import sqlite3
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from gnn import MoleculeGDLModel, molecule_to_graph

# ---------------------------------------------------------------
# Hard-coded elite seed set: high-quality curated references
# always included regardless of SQLite sampling.
# Format: (SMILES, hERG, Ames, DILI, LD50)
# ---------------------------------------------------------------
SEED_REFERENCE_SET = [
    # hERG cardiotoxins
    ("O=C(c1ccc(F)cc1)CCCN2CCC(n3c(=O)[nH]c4ccccc43)CC2", 1, 0, 1, 0),   # Haloperidol
    ("CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4", 1, 0, 1, 0), # Terfenadine
    ("COc1ccc(cc1)C2N(CCN2CCc3ccc(F)cc3)c4ccc(F)cc4", 1, 0, 0, 0),        # Astemizole
    ("CN1CCN(CC1)C2=Nc3cc(Cl)ccc3Nc4ccccc42", 1, 0, 1, 0),                # Clozapine
    # Ames mutagens
    ("O=N(=O)c1ccc(Cl)cc1", 0, 1, 1, 1),  # 1-Chloro-4-nitrobenzene
    ("O=N(=O)c1ccccc1", 0, 1, 0, 0),      # Nitrobenzene
    ("Nc1ccc2ccccc2c1", 0, 1, 1, 0),      # 2-Naphthylamine
    ("Cc1c(N(=O)=O)cc(C)c(N(=O)=O)c1C(C)(C)C", 0, 1, 0, 0),              # Musk xylene
    # DILI hepatotoxins
    ("CC(=O)Nc1ccc(O)cc1", 0, 0, 1, 0),   # Acetaminophen
    ("O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl", 0, 0, 1, 0),  # Diclofenac
    ("CC1Oc2ccc(CC3SC(=O)NC3=O)cc2Oc4c1c(C)c(O)c(C)c4C", 0, 0, 1, 0),    # Troglitazone
    # LD50 acute toxics
    ("CN1C2CCC1C(C(=O)OC)C(OC(=O)c3ccccc3)C2", 0, 0, 0, 1),  # Cocaine
    ("CN1CCCC1c2cccnc2", 0, 0, 0, 1),                          # Nicotine
    ("CC(C)OP(=O)(F)OC(C)C", 0, 0, 0, 1),                     # Sarin
    # Safe controls
    ("CC(=O)Oc1ccccc1C(=O)O", 0, 0, 0, 0),    # Aspirin
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0, 0, 0, 0),  # Ibuprofen
    ("OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O", 0, 0, 0, 0),  # Glucose
    ("O=C(O)CC(O)(C(=O)O)CC(=O)O", 0, 0, 0, 0),  # Citric Acid
    ("CCO", 0, 0, 0, 0),  # Ethanol
    ("CN1C(=O)N(C)c2ncn(C)c2C1=O", 0, 0, 0, 0),  # Caffeine
]

# ---------------------------------------------------------------
# Endpoint column names in the SQLite database
# ---------------------------------------------------------------
ENDPOINT_COLS = ["herg_blocker", "ames_mutagen", "dili_hepatotox", "ld50_high_tox"]

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "failure_database.sqlite")


def load_balanced_dataset_from_db(samples_per_endpoint=60, neg_multiplier=2):
    """
    Loads a class-balanced training set from the SQLite failure database.

    Strategy:
      - For each of the 4 endpoints, sample up to `samples_per_endpoint` positives.
      - Collect a pool of safe (all-zero) negatives of size
        `len(positives) * neg_multiplier`, capped at a max to keep training fast.
      - Merge with the elite seed set (always included).
      - De-duplicate by canonical SMILES.
      - Returns list of (smiles, herg, ames, dili, ld50) tuples.
    """
    if not os.path.exists(DB_PATH):
        print(f"[WARN] SQLite database not found at {DB_PATH}. Falling back to seed set only.")
        return SEED_REFERENCE_SET[:]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    seen_smiles = set()
    records = {}  # canonical_smiles -> [herg, ames, dili, ld50]

    # 1. Seed set always goes in first
    for entry in SEED_REFERENCE_SET:
        smiles, herg, ames, dili, ld50 = entry
        seen_smiles.add(smiles)
        records[smiles] = [herg, ames, dili, ld50]

    # 2. Sample positives for each endpoint
    for col_idx, col in enumerate(ENDPOINT_COLS):
        cur.execute(
            f"SELECT smiles, herg_blocker, ames_mutagen, dili_hepatotox, ld50_high_tox "
            f"FROM failures WHERE {col} = 1 ORDER BY RANDOM() LIMIT ?",
            (samples_per_endpoint,)
        )
        rows = cur.fetchall()
        for row in rows:
            smi, herg, ames, dili, ld50 = row
            if smi and smi not in seen_smiles:
                seen_smiles.add(smi)
                records[smi] = [herg, ames, dili, ld50]

    # 3. Sample negatives (all endpoints = 0, i.e. structurally safe compounds)
    n_positives = len(records) - len(SEED_REFERENCE_SET)
    n_negatives = max(len(SEED_REFERENCE_SET) * 2, n_positives * neg_multiplier)
    n_negatives = min(n_negatives, 400)  # Cap to keep training fast

    cur.execute(
        "SELECT smiles FROM failures "
        "WHERE herg_blocker=0 AND ames_mutagen=0 AND dili_hepatotox=0 AND ld50_high_tox=0 "
        "AND failed_clinical_trial=0 "
        "ORDER BY RANDOM() LIMIT ?",
        (n_negatives,)
    )
    for (smi,) in cur.fetchall():
        if smi and smi not in seen_smiles:
            seen_smiles.add(smi)
            records[smi] = [0, 0, 0, 0]

    conn.close()

    dataset = [(smi, *labels) for smi, labels in records.items()]
    random.shuffle(dataset)
    return dataset


def train_and_save():
    print("=" * 60)
    print("ToxGDL Database-Driven GNN Training Pipeline")
    print("=" * 60)

    dataset = load_balanced_dataset_from_db(samples_per_endpoint=60, neg_multiplier=2)
    print(f"\nLoaded {len(dataset)} compounds from curated database + seed set.")

    # Count endpoint positives
    herg_pos = sum(1 for r in dataset if r[1] == 1)
    ames_pos = sum(1 for r in dataset if r[2] == 1)
    dili_pos = sum(1 for r in dataset if r[3] == 1)
    ld50_pos = sum(1 for r in dataset if r[4] == 1)
    neg_only = sum(1 for r in dataset if r[1] == 0 and r[2] == 0 and r[3] == 0 and r[4] == 0)
    print(f"  hERG positives : {herg_pos}")
    print(f"  Ames positives : {ames_pos}")
    print(f"  DILI positives : {dili_pos}")
    print(f"  LD50 positives : {ld50_pos}")
    print(f"  Safe negatives : {neg_only}")

    print("\nBuilding molecular graphs...")
    graphs = []
    labels = []
    skipped = 0

    for entry in dataset:
        smiles, herg, ames, dili, ld50 = entry
        graph = molecule_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
            labels.append([float(herg), float(ames), float(dili), float(ld50)])
        else:
            skipped += 1

    print(f"Successfully built {len(graphs)} molecular graphs. ({skipped} skipped)")

    if len(graphs) == 0:
        print("ERROR: No valid molecules found. Aborting training.")
        return

    # -------------------------------------------------------
    # Class-weighted BCE loss to compensate for imbalance.
    # We compute positive_weight for each endpoint:
    #   pos_weight[i] = #negatives / #positives
    # -------------------------------------------------------
    labels_arr = np.array(labels)
    pos_weights = []
    for i in range(4):
        n_pos = max(1, int(labels_arr[:, i].sum()))
        n_neg = max(1, len(labels_arr) - n_pos)
        pos_weights.append(n_neg / n_pos)

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"\nClass pos_weights: hERG={pos_weights[0]:.1f} | Ames={pos_weights[1]:.1f} | "
          f"DILI={pos_weights[2]:.1f} | LD50={pos_weights[3]:.1f}")

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # -------------------------------------------------------
    # Model, optimizer, weighted criterion
    # -------------------------------------------------------
    model = MoleculeGDLModel(num_atom_feats=31, num_bond_feats=6, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    epochs = 200
    print(f"\nTraining GDL MPNN for {epochs} epochs on {len(graphs)} real compounds...")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Shuffle molecule order each epoch
        indices = list(range(len(graphs)))
        random.shuffle(indices)

        for i in indices:
            optimizer.zero_grad()
            graph = graphs[i]
            x, edge_index, edge_attr = graph["x"], graph["edge_index"], graph["edge_attr"]
            y = labels_tensor[i].unsqueeze(0)  # [1, 4]

            pred = model(x, edge_index, edge_attr)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            probs = torch.sigmoid(pred)
            preds_bin = (probs > 0.5).float()
            correct += (preds_bin == y).sum().item()
            total += 4

        scheduler.step()

        avg_loss = epoch_loss / len(graphs)
        acc = correct / total * 100

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 40 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

    # Save best weights
    os.makedirs(os.path.join(os.path.dirname(__file__), "weights"), exist_ok=True)
    weight_path = os.path.join(os.path.dirname(__file__), "weights", "gdl_toxicity_model.pt")
    torch.save(best_state, weight_path)
    print(f"\nBest model (loss={best_loss:.4f}) saved to {weight_path}")
    print("Training complete.")


if __name__ == "__main__":
    train_and_save()
