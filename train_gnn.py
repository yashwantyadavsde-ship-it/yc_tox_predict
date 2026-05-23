import torch
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from graph_builder import smiles_to_graph

# 1. Define the Graph Neural Network Architecture
class ToxicityGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ToxicityGNN, self).__init__()
        torch.manual_seed(12345)
        # We use 1 as input channel because our node feature is just atomic number
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1) # Output 1 value for binary classification (Toxic or Safe)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer (Pool node features to create a graph-level feature)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def main():
    print("Loading and preprocessing data...")
    csv_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    df = pd.read_csv(csv_url)
    
    # Drop rows where CT_TOX is NaN
    df = df.dropna(subset=['CT_TOX'])
    
    data_list = []
    toxic_count = 0
    safe_count = 0
    
    for idx, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'], row['CT_TOX'])
        if graph is not None and graph.num_nodes > 0:
            data_list.append(graph)
            if row['CT_TOX'] == 1:
                toxic_count += 1
            else:
                safe_count += 1
                
    print(f"Total valid graphs: {len(data_list)} (Safe: {safe_count}, Toxic: {toxic_count})")
    
    # Handle Class Imbalance with weighted loss
    # Since safe drugs heavily outnumber toxic ones, we tell the model to pay more attention to toxic drugs
    pos_weight = torch.tensor([safe_count / toxic_count])
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    # Split dataset into training and testing
    torch.manual_seed(12345)
    data_list = [data_list[i] for i in torch.randperm(len(data_list))]
    split_idx = int(len(data_list) * 0.8)
    train_dataset = data_list[:split_idx]
    test_dataset = data_list[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ToxicityGNN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)

    def test(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in loader:
                out = model(data.x, data.edge_index, data.batch)
                # Apply sigmoid since output is logits
                probs = torch.sigmoid(out).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(data.y.cpu().numpy())
                
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(all_labels, all_preds)
        # Calculate Accuracy (threshold = 0.5)
        preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_labels, preds_binary)
        return roc_auc, acc

    print("\nStarting Training...")
    for epoch in range(1, 31):
        loss = train()
        train_auc, train_acc = test(train_loader)
        test_auc, test_acc = test(test_loader)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} | '
                  f'Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')

    print("\nTraining Complete! MVP Model is ready.")

if __name__ == '__main__':
    main()
