import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import FraudDetector

def load_graph_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_table     = pd.read_csv(os.path.join(base, "data", "node_features.csv"))
    transactions_table = pd.read_csv(os.path.join(base, "data", "transactions.csv"))

    feature_columns = [
        "out_degree", "in_degree", "total_tx",
        "total_out_amount", "total_in_amount",
        "avg_amount", "max_amount", "unique_counterparties",
        "is_offshore", "is_business", "structuring_flag"
    ]

    raw_features  = features_table[feature_columns].values.astype(np.float32)
    normalizer    = StandardScaler()
    raw_features  = normalizer.fit_transform(raw_features)
    labels        = features_table["is_fraud"].values.astype(np.int64)

    all_account_ids      = features_table["account_id"].tolist()
    account_id_to_number = {aid: i for i, aid in enumerate(all_account_ids)}

    edge_list = []
    for _, tx in transactions_table.iterrows():
        sender_number   = account_id_to_number.get(tx["sender_id"])
        receiver_number = account_id_to_number.get(tx["receiver_id"])
        if sender_number is not None and receiver_number is not None:
            edge_list.append([sender_number, receiver_number])

    edge_connections = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    features_tensor  = torch.tensor(raw_features, dtype=torch.float)
    labels_tensor    = torch.tensor(labels, dtype=torch.long)

    graph_data = Data(x=features_tensor, edge_index=edge_connections, y=labels_tensor)
    return graph_data, normalizer, all_account_ids


def train_model(graph_data):
    total_accounts = graph_data.num_nodes
    shuffled_order = torch.randperm(total_accounts)
    train_accounts = shuffled_order[:int(0.8 * total_accounts)]
    test_accounts  = shuffled_order[int(0.8 * total_accounts):]

    train_mask = torch.zeros(total_accounts, dtype=torch.bool)
    test_mask  = torch.zeros(total_accounts, dtype=torch.bool)
    train_mask[train_accounts] = True
    test_mask[test_accounts]   = True

    fraud_model   = FraudDetector(graph_data.num_node_features)
    optimizer     = torch.optim.Adam(fraud_model.parameters(), lr=0.01, weight_decay=5e-4)
    fraud_weight  = (graph_data.y == 0).sum().float() / (graph_data.y == 1).sum().float()
    loss_weights  = torch.tensor([1.0, fraud_weight])
    loss_function = torch.nn.CrossEntropyLoss(weight=loss_weights)

    print("Training fraud detection model...")
    fraud_model.train()
    for epoch_number in range(200):
        optimizer.zero_grad()
        predictions  = fraud_model(graph_data.x, graph_data.edge_index)
        current_loss = loss_function(predictions[train_mask], graph_data.y[train_mask])
        current_loss.backward()
        optimizer.step()

        if (epoch_number + 1) % 50 == 0:
            correct = (predictions[train_mask].argmax(dim=1) == graph_data.y[train_mask]).float().mean()
            print(f"  Epoch {epoch_number+1}/200 - Loss: {current_loss.item():.4f} | Accuracy: {correct:.4f}")

    fraud_model.eval()
    with torch.no_grad():
        predictions = fraud_model(graph_data.x, graph_data.edge_index)
        test_preds  = predictions[test_mask].argmax(dim=1)
        true_labels = graph_data.y[test_mask]
        print("\nTest Results:")
        print(classification_report(true_labels.numpy(), test_preds.numpy(),
              target_names=["Legit", "Fraud"], zero_division=0))

    return fraud_model


if __name__ == "__main__":
    graph_data, normalizer, all_account_ids = load_graph_data()
    print(f"Graph loaded: {graph_data.num_nodes} accounts, {graph_data.edge_index.shape[1]} transactions")
    print(f"Fraud accounts: {graph_data.y.sum().item()}")
    fraud_model = train_model(graph_data)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.save(fraud_model.state_dict(), os.path.join(base, "model", "fraud_gnn.pt"))
    print("\nModel saved to model/fraud_gnn.pt")