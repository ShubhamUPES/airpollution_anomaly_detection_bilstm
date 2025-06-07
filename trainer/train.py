import torch
import numpy as np
import pandas as pd
from models.bilstm_attention import PM25Predictor
from utils.preprocessing import get_normal_data, prepare_city_data
from utils.utils import evaluate_model

def train_model(model, X_train, y_train, epochs=5, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

def compute_threshold(model, X_train, y_train, percentile=99):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_train))
        errors = torch.abs(preds - torch.FloatTensor(y_train)).squeeze().numpy()
    return np.percentile(errors, percentile), errors

def detect_anomalies(model, X_test, y_test, threshold):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test))
        errors = torch.abs(preds - torch.FloatTensor(y_test)).squeeze().numpy()
    return (errors > threshold), errors

def train_per_city(df, window_size=24):
    results = []
    metrics = []

    for City in df['City'].unique():
        print(f"\n🔍 Processing city: {City}")
        df_City = df[df['City'] == City].copy().reset_index(drop=True)
        normal_df = get_normal_data(df_City)
        X_train, y_train, scaler = prepare_city_data(normal_df, window_size)
        X_full, y_full, _ = prepare_city_data(df_City, window_size)

        X_train = X_train.reshape(-1, window_size, 1)
        y_train = y_train.reshape(-1, 1)
        X_full = X_full.reshape(-1, window_size, 1)
        y_full = y_full.reshape(-1, 1)

        model = PM25Predictor()
        train_model(model, X_train, y_train)

        threshold, _ = compute_threshold(model, X_train, y_train)
        anomaly_flags, errors = detect_anomalies(model, X_full, y_full, threshold)

        df_result = df_City.iloc[window_size:].copy().reset_index(drop=True)
        df_result['error'] = errors
        df_result['anomaly'] = anomaly_flags

        results.append(df_result)
        metrics.append(evaluate_model(anomaly_flags, df_result.get("label", None), City))

    return pd.concat(results), metrics
