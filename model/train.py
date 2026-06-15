import torch
import torch.nn as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from gnn_model import FraudDetector

def load_graph_data():
    account_table= pd.read_csv("data/accounts.csv")
    nodeFeature_table= pd.read_csv("data/node_features.csv")
    transaction_table= pd.read_csv("data/transactions.csv")

    feature_coloumn= [
        "times_sent", "times_received", "total_transactions",
    "total_sent", "total_received",
    "average_amount", "largest_amount", "unique_people_dealt_with",
    "is_offshore", "is_business", "structuring_count"
    ]

    raw_Feature= nodeFeature_table[feature_coloumn].values.astype(np.float32)

    normalizer= StandardScaler()
    raw_Feature= normalizer.fit_transform(raw_Feature)

    