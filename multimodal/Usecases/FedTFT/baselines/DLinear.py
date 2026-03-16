import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- pieces from original DLinear ----------
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))       # [B, C, L] -> pooled
        x = x.permute(0, 2, 1)                 # [B, L, C]
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  # seasonal, trend
# -------------------------------------------------

class DLinearClassifier(nn.Module):
    """
    DLinear backbones for binary multi-label classification.
    Expects:
      - seq_x:   [B, seq_len, n_vars]
      - static:  [B, static_dim]
    Returns:
      - logits:  [B, out_dim]   (use BCEWithLogitsLoss)
    """
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        static_dim: int,
        out_dim: int = 3,
        kernel_size: int = 25,
        individual: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.n_vars     = n_vars
        self.static_dim = static_dim
        self.output_dim = out_dim  # used by your server code

        # DLinear decomposition
        self.decomp = series_decomp(kernel_size)

        self.individual = individual
        if individual:
            # Per-channel linear projections seq_len -> 1
            self.lin_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, 1) for _ in range(n_vars)]
            )
            self.lin_trend = nn.ModuleList(
                [nn.Linear(seq_len, 1) for _ in range(n_vars)]
            )
            # init like original DLinear: start as averaging filters
            for i in range(n_vars):
                with torch.no_grad():
                    self.lin_seasonal[i].weight.fill_(1.0 / seq_len)
                    self.lin_trend[i].weight.fill_(1.0 / seq_len)
        else:
            # Shared across channels; we’ll apply them over the last dim after reshape
            self.lin_seasonal = nn.Linear(seq_len, 1)
            self.lin_trend    = nn.Linear(seq_len, 1)
            with torch.no_grad():
                self.lin_seasonal.weight.fill_(1.0 / seq_len)
                self.lin_trend.weight.fill_(1.0 / seq_len)

        # Light norm on per-channel summary then fuse with static
        self.bn_channels = nn.BatchNorm1d(n_vars)
        fused_in = n_vars + static_dim

        self.head = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _project_per_channel(self, x_cl):  # x_cl: [B, C, L]
        if self.individual:
            # Stack per-channel linear projections
            # result: [B, C, 1]
            seas = torch.stack(
                [self.lin_seasonal[i](x_cl[:, i, :]) for i in range(self.n_vars)],
                dim=1,
            )
            trnd = torch.stack(
                [self.lin_trend[i](x_cl[:, i, :]) for i in range(self.n_vars)],
                dim=1,
            )
        else:
            B, C, L = x_cl.shape
            x_flat = x_cl.reshape(B * C, L)           # [B*C, L]
            seas = self.lin_seasonal(x_flat).reshape(B, C, 1)
            trnd = self.lin_trend(x_flat).reshape(B, C, 1)
        return seas + trnd  # [B, C, 1]

    def forward(self, static_x, seq_x):
        """
        static_x: [B, static_dim]
        seq_x:    [B, seq_len, n_vars]
        """
        # DLinear seasonal/trend decomposition
        seasonal, trend = self.decomp(seq_x)         # [B, L, C], [B, L, C]
        # operate as channel-last → channel-first for linear over time
        seas_cf = seasonal.permute(0, 2, 1)          # [B, C, L]
        trnd_cf = trend.permute(0, 2, 1)             # [B, C, L]

        # project seq_len → 1 for each channel (seasonal + trend)
        y_seas = self._project_per_channel(seas_cf)  # [B, C, 1]
        y_trnd = self._project_per_channel(trnd_cf)  # [B, C, 1]
        chan_summary = (y_seas + y_trnd).squeeze(-1) # [B, C]

        # normalize channel summary and fuse with static
        chan_summary = self.bn_channels(chan_summary)
        fused = torch.cat([chan_summary, static_x], dim=1)  # [B, C + static_dim]

        # logits for each label (no sigmoid here)
        logits = self.head(fused)                   # [B, out_dim]
        return logits
import json
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
