import pandas as pd
import networkx as nx
import numpy as np

def load_data():
    accounts = pd.read_csv("data/accounts.csv")
    transactions = pd.read_csv("data/transactions.csv")
    return accounts, transactions

def build_graph(accounts, transactions):
    G = nx.DiGraph()

    # Add nodes (accounts) with features
    for _, acc in accounts.iterrows():
        G.add_node(acc["account_id"], 
            name=acc["name"],
            account_type=acc["account_type"],
            country=acc["country"],
            is_fraud=acc["is_fraud"]
        )

    # Add edges (transactions) with features
    for _, tx in transactions.iterrows():
        G.add_edge(
            tx["sender_id"],
            tx["receiver_id"],
            transaction_id=tx["transaction_id"],
            amount=tx["amount"],
            tx_type=tx["tx_type"],
            is_fraud=tx["is_fraud"],
            flag_reason=tx["flag_reason"]
        )

    return G

def compute_node_features(G, accounts):
    features = []

    for _, acc in accounts.iterrows():
        node = acc["account_id"]

        # Outgoing transactions
        out_edges = list(G.out_edges(node, data=True))
        out_amounts = [e[2]["amount"] for e in out_edges]

        # Incoming transactions  
        in_edges = list(G.in_edges(node, data=True))
        in_amounts = [e[2]["amount"] for e in in_edges]

        all_amounts = out_amounts + in_amounts

        # Feature engineering — what the GNN learns from
        feat = {
            "account_id": node,
            "is_fraud": acc["is_fraud"],
            "out_degree": len(out_edges),
            "in_degree": len(in_edges),
            "total_tx": len(out_edges) + len(in_edges),
            "total_out_amount": sum(out_amounts) if out_amounts else 0,
            "total_in_amount": sum(in_amounts) if in_amounts else 0,
            "avg_amount": np.mean(all_amounts) if all_amounts else 0,
            "max_amount": max(all_amounts) if all_amounts else 0,
            "unique_counterparties": len(set(
                [e[1] for e in out_edges] + [e[0] for e in in_edges]
            )),
            "is_offshore": 1 if acc["country"] not in ["US", "UK"] else 0,
            "is_business": 1 if acc["account_type"] == "business" else 0,
            "structuring_flag": sum(
                1 for a in out_amounts if 9000 <= a <= 9999
            ),
        }
        features.append(feat)

    return pd.DataFrame(features)

def print_graph_stats(G, features_df):
    print(f"Nodes (accounts):     {G.number_of_nodes()}")
    print(f"Edges (transactions): {G.number_of_edges()}")
    print(f"Fraud nodes:          {features_df['is_fraud'].sum()}")
    print(f"\nTop 5 most connected accounts:")
    degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    for node, deg in degree:
        fraud = G.nodes[node].get("is_fraud", 0)
        print(f"  {node} — {deg} connections {'⚠ FRAUD' if fraud else ''}")

if __name__ == "__main__":
    print("Loading data...")
    accounts, transactions = load_data()

    print("Building graph...")
    G = build_graph(accounts, transactions)

    print("Computing node features...")
    features_df = compute_node_features(G, accounts)

    print_graph_stats(G, features_df)

    # Save features for the GNN
    features_df.to_csv("data/node_features.csv", index=False)
    print(f"\nNode features saved to data/node_features.csv")
    print(f"Feature columns: {[c for c in features_df.columns if c not in ['account_id','is_fraud']]}")