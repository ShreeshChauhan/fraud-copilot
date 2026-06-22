from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
import sys
import os

base_dir = os.path.dirname(__file__)
for folder in ["model", "llm", "graph"]:
    sys.path.insert(0, os.path.join(base_dir, "..", folder))

from gnn_model import FraudDetector
from explainer import generate_risk_explanation, answer_investigator_question, get_account_summary
from visualize import render_network_html
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI(title="Fraud Investigation Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Define BASE path and load data once at startup ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_columns = [
    "out_degree", "in_degree", "total_tx",
    "total_out_amount", "total_in_amount",
    "avg_amount", "max_amount", "unique_counterparties",
    "is_offshore", "is_business", "structuring_flag"
]

features_table = pd.read_csv(os.path.join(BASE, "data", "node_features.csv"))
raw_features   = features_table[feature_columns].values.astype(np.float32)
normalizer     = StandardScaler()
raw_features   = normalizer.fit_transform(raw_features)

model = FraudDetector(in_channels=len(feature_columns))
model.load_state_dict(torch.load(os.path.join(BASE, "model", "fraud_gnn.pt"),
                      weights_only=True))
model.eval()
# ────────────────────────────────────────────────────


def get_risk_score(account_id):
    transactions = pd.read_csv(os.path.join(BASE, "data", "transactions.csv"))
    all_account_ids = features_table["account_id"].tolist()
    if account_id not in all_account_ids:
        return None
    account_id_to_number = {aid: i for i, aid in enumerate(all_account_ids)}
    edge_list = []
    for _, tx in transactions.iterrows():
        s = account_id_to_number.get(tx["sender_id"])
        r = account_id_to_number.get(tx["receiver_id"])
        if s is not None and r is not None:
            edge_list.append([s, r])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(raw_features, dtype=torch.float)
    with torch.no_grad():
        out   = model(x, edge_index)
        probs = torch.softmax(out, dim=1)
    idx = account_id_to_number[account_id]
    return round(probs[idx][1].item(), 4)


@app.get("/")
def root():
    return {"status": "Fraud Investigation Copilot API is running"}


@app.get("/account/{account_id}/risk")
def account_risk(account_id: str):
    score = get_risk_score(account_id)
    if score is None:
        return {"error": "Account not found"}
    return {"account_id": account_id, "risk_score": score}


@app.get("/account/{account_id}/explanation")
def account_explanation(account_id: str):
    score = get_risk_score(account_id)
    if score is None:
        return {"error": "Account not found"}