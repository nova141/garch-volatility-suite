import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from core.utils import create_sequences, scale_data
import statsmodels.api as sm


class GARCH_GRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1, output_size=1):
        super(GARCH_GRU, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.omega, self.alpha, self.beta = nn.Parameter(torch.rand(1)), nn.Parameter(torch.rand(1)), nn.Parameter(
            torch.rand(1))
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_return):
        batch_size, seq_len = x_return.shape
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32).to(x_return.device)
        sigma2 = torch.zeros(batch_size, 1, dtype=torch.float32).to(x_return.device)
        for t in range(seq_len):
            error_sq = (x_return[:, t].unsqueeze(1)) ** 2
            sigma2 = self.omega + self.alpha * error_sq + self.beta * sigma2
            combined_input = torch.cat((x_return[:, t].unsqueeze(1), sigma2), dim=1)
            _, h0 = self.gru(combined_input.unsqueeze(1), h0)
        return self.fc(h0.squeeze(0))

@st.cache_resource(ttl="1h")
def train_hybrid_model(_model, log_returns, sequence_length=10, epochs=100, lr=0.001):
    returns_np = log_returns.values.astype(np.float32)
    X_returns, y_vol = create_sequences(pd.Series(returns_np), sequence_length)
    _, y_vol = create_sequences(pd.Series(returns_np ** 2), sequence_length)
    X_scaled, _ = scale_data(X_returns)
    y_scaled, vol_scaler = scale_data(y_vol.reshape(-1, 1))
    X_tensor, y_tensor = torch.from_numpy(X_scaled).float(), torch.from_numpy(y_scaled).float()

    model, criterion, optimizer = _model, nn.MSELoss(), torch.optim.Adam(_model.parameters(), lr=lr)
    loss_history, param_history = [], {'omega': [], 'alpha': [], 'beta': []}
    progress_bar, status_text = st.progress(0), st.empty()
    for epoch in range(epochs):
        model.train()
        loss = criterion(model(X_tensor), y_tensor)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        loss_history.append(loss.item())
        for p in param_history: param_history[p].append(model.state_dict()[p].item())
        progress_bar.progress((epoch + 1) / epochs);
        status_text.text(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}")
    status_text.success("Training complete!")
    return model, vol_scaler, loss_history, param_history


def predict_with_hybrid_model(model, log_returns, sequence_length, vol_scaler):
    returns_np = log_returns.values.astype(np.float32)
    X_returns, _ = create_sequences(pd.Series(returns_np), sequence_length)
    X_scaled, _ = scale_data(X_returns)
    X_tensor = torch.from_numpy(X_scaled).float()
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_tensor)
    return pd.Series(vol_scaler.inverse_transform(preds_scaled.numpy()).flatten(),
                     index=log_returns.index[sequence_length:])


def run_mincer_zarnowitz_regression(realized_vol, forecast_vol):
    df = pd.concat([realized_vol, forecast_vol], axis=1).dropna()
    df.columns = ['realized', 'forecast']
    X = sm.add_constant(df['forecast'])
    model = sm.OLS(df['realized'], X).fit()
    return {'intercept': model.params['const'], 'slope': model.params['forecast'], 'r_squared': model.rsquared,
            'plot_data': df, 'fitted_values': model.fittedvalues}


def monte_carlo_forecast_cone(model, last_return_sequence, last_sigma2, vol_scaler, horizon=63, simulations=500):
    model.eval()
    all_paths = []
    for _ in range(simulations):
        path = []
        h_init = torch.zeros(model.num_layers, 1, model.hidden_size, dtype=torch.float32)
        sigma2_init = torch.tensor([[last_sigma2]], dtype=torch.float32)

        for t in range(last_return_sequence.shape[1]):
            error_sq = last_return_sequence[:, t].unsqueeze(1) ** 2
            sigma2_init = model.omega + model.alpha * error_sq + model.beta * sigma2_init
            combined_input = torch.cat((last_return_sequence[:, t].unsqueeze(1), sigma2_init), dim=1).unsqueeze(1)
            _, h_init = model.gru(combined_input, h_init)

        current_h, current_sigma2 = h_init, sigma2_init
        with torch.no_grad():
            for _ in range(horizon):
                output = model.fc(current_h.squeeze(0))
                path.append(output.item())
                forecasted_vol = np.sqrt(max(output.item(), 0))
                # --- FIX: Ensure new random tensors are also float32 ---
                random_shock = torch.tensor([[np.random.normal(0, forecasted_vol)]], dtype=torch.float32)
                error_sq = random_shock ** 2
                current_sigma2 = model.omega + model.alpha * error_sq + model.beta * current_sigma2
                combined_input = torch.cat((random_shock, current_sigma2), dim=1).unsqueeze(1)
                _, current_h = model.gru(combined_input, current_h)
        all_paths.append(path)

    paths_unscaled = vol_scaler.inverse_transform(np.array(all_paths).T).T
    return np.median(paths_unscaled, axis=0), np.percentile(paths_unscaled, 5, axis=0), np.percentile(paths_unscaled,
                                                                                                      95, axis=0)


def calculate_saliency(model, input_sequence_scaled):
    model.eval()
    input_sequence_scaled.requires_grad_(True)
    prediction = model(input_sequence_scaled)
    if input_sequence_scaled.grad is not None:
        input_sequence_scaled.grad.zero_()
    prediction.sum().backward()
    return input_sequence_scaled.grad.abs().squeeze().numpy()
