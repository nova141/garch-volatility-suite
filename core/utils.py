import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def create_sequences(data: pd.Series, sequence_length: int):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:(i + sequence_length)].values
        target = data.iloc[i + sequence_length]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def scale_data(data: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape data for scaler: from (n_samples, n_features) to (n_samples * n_features, 1)
    reshaped_data = data.reshape(-1, 1)
    scaled_data = scaler.fit_transform(reshaped_data)
    # Reshape back to original shape
    scaled_data = scaled_data.reshape(data.shape)
    return scaled_data, scaler
