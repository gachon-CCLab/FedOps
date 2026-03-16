# fedformer_classifier.py
import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock
from layers.MultiWaveletCorrelation import MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
import json

class FedformerClassifier(nn.Module):
    """
    FEDformer-as-a-classifier:
      • Encoder-only FEDformer backbone (Fourier or Wavelets)
      • Static feature MLP
      • Fusion head -> logits (B, out_dim)

    Forward signature matches your FL code:
        logits = model(static_x, seq_x)  # (B, out_dim)
    """

    def __init__(
        self,
        *,
        seq_len: int,
        n_vars: int,
        static_dim: int,
        d_model: int = 128,
        n_heads: int = 8,           # keep 8 (your Fourier/Wavelet layers assume 8)
        e_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        version: str = "Wavelets",  # "Wavelets" or "Fourier"
        modes: int = 32,
        moving_avg=(12, 24),
        L: int = 1,                 # wavelet level
        base: str = "legendre",
        out_dim: int = 3,           # 3 horizons -> 3 logits
        pool: str = "mean",         # "mean" or "last"
    ):
        super().__init__()
        self.output_dim = out_dim
        self.seq_len = seq_len
        self.n_vars  = n_vars
        self.d_model = d_model
        self.n_heads = n_heads

        # No positional embedding; use timeF (we'll pass zeros as time marks)
        self.enc_embedding = DataEmbedding_wo_pos(
            c_in=n_vars, d_model=d_model, embed_type="timeF", freq="h", dropout=dropout
        )

        # Choose FEDformer correlation block
        if version.lower().startswith("wave"):
            enc_corr = MultiWaveletTransform(ich=d_model, L=L, base=base, attention_dropout=dropout)
        else:
            enc_corr = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=seq_len, modes=modes)

        # Encoder (no decoder needed for classification)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(enc_corr, d_model=d_model, n_heads=n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    moving_avg=list(moving_avg) if not isinstance(moving_avg, list) else moving_avg,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        # Static branch + fusion head
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(d_model + d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, static_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        """
        static_x: (B, static_dim)
        seq_x:    (B, seq_len, n_vars)
        returns:  (B, out_dim) logits
        """
        B, L, V = seq_x.shape
        assert L == self.seq_len and V == self.n_vars, f"Expected ({self.seq_len},{self.n_vars}), got ({L},{V})"

        device = seq_x.device
        # DataEmbedding_wo_pos with embed_type="timeF" ('h') expects 4-dim time mark.
        x_mark = torch.zeros(B, L, 4, device=device)

        enc_in       = self.enc_embedding(seq_x, x_mark)  # (B, L, d_model)
        enc_out, _   = self.encoder(enc_in)               # (B, L, d_model)
        ts_repr      = enc_out[:, -1, :] if self.pool == "last" else enc_out.mean(dim=1)
        st_repr      = self.static_proj(static_x)
        fused        = torch.cat([ts_repr, st_repr], dim=-1)
        logits       = self.head(fused)
        return logits


if __name__ == "__main__":
    # quick shape test
    m = FedformerClassifier(seq_len=192, n_vars=25, static_dim=14, out_dim=3).eval()
    sx   = torch.randn(2, 14)
    seqx = torch.randn(2, 192, 25)
    out  = m(sx, seqx)
    print(out.shape)  # torch.Size([2, 3])

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