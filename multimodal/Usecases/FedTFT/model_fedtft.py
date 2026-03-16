import torch
import torch.nn as nn
import pytorch_lightning as pl
import json

# ----------------------------
# Helper Modules
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
    def __init__(self, input_size, hidden_size, output_size, dropout, context_size=None, is_temporal=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_temporal = is_temporal

        if self.is_temporal:
            if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size, self.output_size))

            if self.context_size is not None:
                self.c = TemporalLayer(nn.Linear(self.context_size, self.hidden_size, bias=False))

            self.dense1 = TemporalLayer(nn.Linear(self.input_size, self.hidden_size))
            self.elu = nn.ELU()
            self.dense2 = TemporalLayer(nn.Linear(self.hidden_size, self.output_size))
            self.dropout_layer = nn.Dropout(self.dropout)
            self.gate = TemporalLayer(GLU(self.output_size))
            self.layer_norm = nn.LayerNorm(self.output_size)
        else:
            if self.input_size != self.output_size:
                self.skip_layer = nn.Linear(self.input_size, self.output_size)

            if self.context_size is not None:
                self.c = nn.Linear(self.context_size, self.hidden_size, bias=False)

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
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)
        return attn_output, attn_weights

class TFTPredictor(pl.LightningModule):
    def __init__(self, input_dim, static_dim, hidden_dim, output_dim, dropout=0.1, learning_rate=1e-3, num_heads=4):
        super().__init__()
        self.static_grn = GateResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout, is_temporal=False)
        self.variable_selector = GateResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.seq_transform = GateResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.grn1 = GateResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout, is_temporal=True)
        self.grn2 = GateResidualNetwork(hidden_dim, hidden_dim, output_dim, dropout, is_temporal=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, static_x, seq_x, return_attn=False):
        static_features = self.static_grn(static_x)  # [B, D]
        variable_scores = self.variable_selector(seq_x)  # [B, T, D]
        seq_features = self.seq_transform(seq_x) * variable_scores  # [B, T, D]
        lstm_out, _ = self.lstm(seq_features)  # [B, T, D]

        # Cross-attention: each timestep attends to static features
        static_context = static_features.unsqueeze(1).repeat(1, lstm_out.size(1), 1)  # [B, T, D]
        attention_output, attn_weights = self.multihead_attention(
            query=lstm_out,
            key=static_context,
            value=static_context
        )

        combined = attention_output[:, -1, :]  # Already contains static info via attention
        x = self.grn1(combined.unsqueeze(1))
        x = self.dropout_layer(x)
        x = self.grn2(x).squeeze(1)

        if return_attn:
            return x, attn_weights
        return x

    def training_step(self, batch, batch_idx):
        static_x, seq_x, y = batch
        preds = self(static_x, seq_x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        static_x, seq_x, y = batch
        preds = self(static_x, seq_x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TFTDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        static_x, seq_x, y = self.sequences[idx]
        static_x = torch.tensor(static_x, dtype=torch.float32)
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return static_x, seq_x, y

def load_hospital_mapping():
    with open("hospital_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)
