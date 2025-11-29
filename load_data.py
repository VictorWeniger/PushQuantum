import numpy as np
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# Parsing helpers
def parse_surface_metadata(df):
    tenors, maturities = [], []
    mapping = {}
    for col in df.columns:
        if col == 'Date':
            continue
        parts = col.split(';')
        tenor = float(parts[0].split(':')[1].strip())
        maturity = float(parts[1].split(':')[1].strip())
        tenors.append(tenor)
        maturities.append(maturity)
        mapping[col] = (tenor, maturity)
    return sorted(set(tenors)), sorted(set(maturities)), mapping

def surface_for_date(df, idx, unique_tenors, unique_maturities, mapping):
    row = df.iloc[idx]
    surface = np.full((len(unique_tenors), len(unique_maturities)), np.nan)
    for col, (t, m) in mapping.items():
        t_idx = unique_tenors.index(t)
        m_idx = unique_maturities.index(m)
        surface[t_idx, m_idx] = row[col]
    return surface


## real data

PROJECT_ROOT = "./provided"

DATA_PATH = os.path.join(PROJECT_ROOT, "train.xlsx")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Using project root: {PROJECT_ROOT}")
print(f"Train data path: {DATA_PATH}")
print(f"Plots will be saved to: {PLOTS_DIR}")

df = pd.read_excel(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

# Keep Date as first column
df = df[['Date'] + [c for c in df.columns if c != 'Date']]
feature_cols = [c for c in df.columns if c != 'Date']

unique_tenors, unique_maturities, tm_map = parse_surface_metadata(df)

window_in = 6
window_out = 5

S = len(df) - window_in - window_out

X = np.zeros((S, window_in, len(unique_tenors), len(unique_maturities), ), dtype="float")
y = np.zeros((S, window_out, len(unique_tenors), len(unique_maturities), ), dtype="float")

for s in range(len(df) - window_in - window_out):
    for col, (t, m) in tm_map.items():

        for i in range(0, window_in):
            surface = np.full((len(unique_tenors), len(unique_maturities)), np.nan)
            t_idx = unique_tenors.index(t)
            m_idx = unique_maturities.index(m)
            X[s, i, t_idx, m_idx] = df[col][s + i]

        for i in range(0, window_out):
            surface = np.full((len(unique_tenors), len(unique_maturities)), np.nan)
            t_idx = unique_tenors.index(t)
            m_idx = unique_maturities.index(m)
            y[s, i, t_idx, m_idx] = df[col][s + window_in + i]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

mean = X_train.mean(dim=0, keepdim=True)
std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size: {X_test.shape[0]} samples")

print(X_train.shape, y_train.shape)