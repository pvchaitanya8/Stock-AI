import torch
import torch.nn as nn


class LSTMAttentionForecaster(nn.Module):
    """
    3-layer LSTM followed by multi-head self-attention for multi-step
    stock price forecasting.  Dropout stays active at inference time so
    the same forward pass can be called repeatedly for MC Dropout
    uncertainty estimation.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
        horizon: int = 30,
    ):
        super().__init__()
        self.horizon = horizon

        self.input_proj = nn.Linear(n_features, hidden)

        # LSTM inter-layer dropout is applied between layers (not after last)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)          # → (batch, seq_len, hidden)
        x, _ = self.lstm(x)             # → (batch, seq_len, hidden)
        x = self.dropout(x)

        attn_out, _ = self.attn(x, x, x)   # self-attention
        x = self.norm(x + attn_out)         # residual + layer-norm

        last = x[:, -1, :]              # take last timestep → (batch, hidden)
        return self.decoder(last)       # → (batch, horizon)
