"""
model_wesad.py — TFT backbone + binary stress head (StressTFT)

Dimensions
----------
  static_dim : 8   (subject-level baseline physiological stats)
  input_dim  : 14  (per-window sensor features)
  seq_len    : 10  (10-minute lookback)
  hidden_dim : 64
"""

import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.a       = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.b       = nn.Linear(input_size, input_size)

    def forward(self, x):
        return torch.mul(self.sigmoid(self.b(x)), self.a(x))


class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        b, t, h = x.size()
        out = self.module(x.contiguous().view(b * t, h))
        return out.view(b, t, out.size(-1))


class GateResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout,
                 context_size=None, is_temporal=True):
        super().__init__()
        self.input_size   = input_size
        self.output_size  = output_size
        self.context_size = context_size

        _L = TemporalLayer if is_temporal else lambda m: m

        if input_size != output_size:
            self.skip_layer = _L(nn.Linear(input_size, output_size))
        if context_size is not None:
            self.c = _L(nn.Linear(context_size, hidden_size, bias=False))

        self.dense1        = _L(nn.Linear(input_size, hidden_size))
        self.elu           = nn.ELU()
        self.dense2        = _L(nn.Linear(hidden_size, output_size))
        self.dropout_layer = nn.Dropout(dropout)
        self.gate          = _L(GLU(output_size))
        self.layer_norm    = nn.LayerNorm(output_size)

    def forward(self, x, c=None):
        a = self.skip_layer(x) if self.input_size != self.output_size else x
        h = self.dense1(x)
        if c is not None and self.context_size is not None:
            h = h + self.c(c.unsqueeze(1))
        h = self.elu(h)
        h = self.dense2(h)
        h = self.dropout_layer(h)
        h = self.gate(h) + a
        return self.layer_norm(h)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn       = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, q, k, v):
        out, w = self.attn(q, k, v)
        out    = self.layer_norm(self.dropout(out) + q)
        return out, w


class StressTFT(nn.Module):
    """TFT backbone with a single binary output head for stress classification."""

    def __init__(self, input_dim=14, static_dim=8, hidden_dim=64,
                 dropout=0.1, num_heads=4):
        super().__init__()
        self.static_grn          = GateResidualNetwork(
            static_dim, hidden_dim, hidden_dim, dropout, is_temporal=False)
        self.variable_selector   = GateResidualNetwork(
            input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.seq_transform       = GateResidualNetwork(
            input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.lstm                = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.grn1                = GateResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.dropout_layer       = nn.Dropout(dropout)
        self.grn2                = GateResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, is_temporal=False)
        self.output_head         = nn.Linear(hidden_dim, 1)

    def forward(self, static_x, seq_x):
        static_feat  = self.static_grn(static_x)
        var_scores   = self.variable_selector(seq_x)
        seq_feat     = self.seq_transform(seq_x) * var_scores
        lstm_out, _  = self.lstm(seq_feat)

        static_ctx   = static_feat.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        attn_out, _  = self.multihead_attention(lstm_out, static_ctx, static_ctx)

        combined     = attn_out[:, -1, :]
        embedding    = self.grn1(combined.unsqueeze(1)).squeeze(1)
        embedding    = self.dropout_layer(embedding)
        features     = self.grn2(embedding)

        return self.output_head(features).squeeze(-1)   # (B,)


class WESADDataset(torch.utils.data.Dataset):
    def __init__(self, static, seqs, targets):
        self.static  = static
        self.seqs    = seqs
        self.targets = targets.squeeze(-1)   # (N, 1) → (N,)

    def __len__(self):
        return len(self.static)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.static[idx],  dtype=torch.float32),
            torch.tensor(self.seqs[idx],    dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )
