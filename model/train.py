import torch
import torch.nn as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from gnn_model import FraudDetector