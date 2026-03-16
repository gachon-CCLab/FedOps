"""
model_fedtft_hdfp.py — FedTFT Model: Horizon-Decoupled Federated Heads

Architecture (novelty over original intermediate fusion FedTFT):
  Backbone (identical to original):
    static_grn → variable_selector × seq_transform → lstm → cross-attention → grn1

  Output (novelty):
    Original: single shared grn2 (64→3) — all 3 horizons share output weights → gradient interference
    FedTFT:   grn2 reconfigured to 64→64 shared feature GRN (no longer the output layer), feeding
              3 separate nn.Linear(64→1) heads, ALL fully federated — horizon-decoupled
              (plain Linear avoids LayerNorm(1) numerical collapse; gradient isolation at output preserved)

  All parameters (backbone + 3 horizon heads) are federated each round.
  Each horizon head receives gradient ONLY from its own horizon's loss → no inter-horizon
  gradient interference. Full federation is preserved for global performance.
"""

import torch
import torch.nn as nn
import json

NUM_HORIZONS = 3  # 1h, 1d, 1w


# ----------------------------
# Helper Modules (identical to model_fedtft.py)
# ----------------------------
class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.a = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.b = nn.Linear(input_size, input_size)

    def forward(self, x):
        gate = self.sigmoid(self.b(x))
        x = self.a(x)
        return torch.mul(gate, x)


class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        b, t, h = x.size()
        flat = x.contiguous().view(b * t, h)
        out = self.module(flat)
        h2 = out.size(-1)
        return out.view(b, t, h2)


class GateResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout,
                 context_size=None, is_temporal=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_temporal = is_temporal

        if self.is_temporal:
            if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(
                    nn.Linear(self.input_size, self.output_size))
            if self.context_size is not None:
                self.c = TemporalLayer(
                    nn.Linear(self.context_size, self.hidden_size, bias=False))
            self.dense1 = TemporalLayer(
                nn.Linear(self.input_size, self.hidden_size))
            self.elu = nn.ELU()
            self.dense2 = TemporalLayer(
                nn.Linear(self.hidden_size, self.output_size))
            self.dropout_layer = nn.Dropout(self.dropout)
            self.gate = TemporalLayer(GLU(self.output_size))
            self.layer_norm = nn.LayerNorm(self.output_size)
        else:
            if self.input_size != self.output_size:
                self.skip_layer = nn.Linear(self.input_size, self.output_size)
            if self.context_size is not None:
                self.c = nn.Linear(
                    self.context_size, self.hidden_size, bias=False)
            self.dense1 = nn.Linear(self.input_size, self.hidden_size)
            self.elu = nn.ELU()
            self.dense2 = nn.Linear(self.hidden_size, self.output_size)
            self.dropout_layer = nn.Dropout(self.dropout)
            self.gate = GLU(self.output_size)
            self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x, c=None):
        if self.input_size != self.output_size:
            a = self.skip_layer(x)
        else:
            a = x
        x = self.dense1(x)
        if c is not None and self.context_size is not None:
            c = self.c(c.unsqueeze(1))
            x += c
        eta_2 = self.elu(x)
        eta_1 = self.dense2(eta_2)
        eta_1 = self.dropout_layer(eta_1)
        gate = self.gate(eta_1)
        gate += a
        x = self.layer_norm(gate)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)
        return attn_output, attn_weights


# ----------------------------
# FedTFT Model: backbone + 3 horizon-decoupled federated heads
# ----------------------------
class TFTPredictor_FedTFT(nn.Module):
    """
    FedTFT: Federated TFT with horizon-decoupled output heads.

    Backbone is identical to the original intermediate fusion TFTPredictor.
    Novelty: the single shared grn2 output head (64→3) is replaced by a reconfigured
    grn2 feature GRN (64→64) followed by 3 separate linear output projections (64→1 each),
    one per prediction horizon, ALL fully federated.

    Each head receives gradient only from its own horizon's loss, eliminating
    inter-horizon gradient interference while preserving full federation.

    Parameters
    ----------
    input_dim  : int  — sequence features (25)
    static_dim : int  — static features (14)
    hidden_dim : int  — hidden dimension (64)
    dropout    : float
    num_heads  : int  — multi-head attention heads
    """

    def __init__(self, input_dim, static_dim, hidden_dim=64,
                 dropout=0.1, num_heads=4):
        super().__init__()
        # ── Backbone (identical to original) ──────────────────────────────
        self.static_grn = GateResidualNetwork(
            static_dim, hidden_dim, hidden_dim, dropout, is_temporal=False)
        self.variable_selector = GateResidualNetwork(
            input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.seq_transform = GateResidualNetwork(
            input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout)
        self.multihead_attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout)
        self.grn1 = GateResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.dropout_layer = nn.Dropout(dropout)

        # ── Shared feature GRN (reconfigured from original grn2) ─────────
        # Original grn2 was (64→3) output head; reconfigured to (64→64) feature transform
        # feeding 3 separate linear output heads — eliminates inter-horizon output interference
        self.grn2 = GateResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, is_temporal=False)

        # ── Horizon-decoupled output projections (ALL fully federated) ────
        # horizon_heads[0]=1h, [1]=1d, [2]=1w
        # Each is a separate nn.Linear(64→1) — avoids LayerNorm(1) collapse bug
        # Gradient for each horizon flows ONLY through its own head
        self.horizon_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(NUM_HORIZONS)
        ])

    def forward(self, static_x, seq_x, return_attn=False):
        # Backbone
        static_features = self.static_grn(static_x)                    # [B, D]
        variable_scores = self.variable_selector(seq_x)                 # [B, T, D]
        seq_features    = self.seq_transform(seq_x) * variable_scores  # [B, T, D]
        lstm_out, _     = self.lstm(seq_features)                       # [B, T, D]

        # Cross-attention: Q=lstm, K=V=static replicated
        static_ctx = static_features.unsqueeze(1).expand(
            -1, lstm_out.size(1), -1)                                   # [B, T, D]
        attn_out, attn_weights = self.multihead_attention(
            query=lstm_out, key=static_ctx, value=static_ctx)

        combined  = attn_out[:, -1, :]                                  # [B, D]
        embedding = self.grn1(combined.unsqueeze(1))                    # [B, 1, D]
        embedding = self.dropout_layer(embedding.squeeze(1))            # [B, D]
        features  = self.grn2(embedding)                                # [B, D]

        # Horizon-decoupled projections: each Linear(64→1) → [B,1] → squeeze → [B]
        logits = torch.cat(
            [head(features) for head in self.horizon_heads],
            dim=1
        )                                                               # [B, 3]

        if return_attn:
            return logits, attn_weights
        return logits


# ----------------------------
# Dataset (identical to model_fedtft.py)
# ----------------------------
class TFTDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        static_x, seq_x, y = self.sequences[idx]
        static_x = torch.tensor(static_x, dtype=torch.float32)
        seq_x    = torch.tensor(seq_x,    dtype=torch.float32)
        y        = torch.tensor(y,         dtype=torch.float32)
        return static_x, seq_x, y


def load_hospital_mapping():
    with open("hospital_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)
