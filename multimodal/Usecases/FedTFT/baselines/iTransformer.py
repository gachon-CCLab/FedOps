# feditransformer_classifier.py
import torch
import torch.nn as nn
import math
from types import SimpleNamespace

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted   # from iTransformer repo

class ITransformerClassifier(nn.Module):
    """
    iTransformer encoder + static-token fusion + 3-logit classifier.
    Inputs:
      static_x: [B, S_static]      (e.g., 14)
      seq_x:    [B, L, N_vars]     (e.g., 192, 25)
    Output:
      logits:   [B, 3]             (1h, 1d, 1w)
    """
    def __init__(
        self,
        seq_len=192,
        n_vars=25,
        static_dim=14,
        d_model=128,
        n_heads=4,
        e_layers=4,
        d_ff=512,
        dropout=0.1,
        use_norm=True,
        out_dim=3,
        add_time_features=True,   # simple sin/cos time markers
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars  = n_vars
        self.use_norm = use_norm
        self.add_time_features = add_time_features

        # --- iTransformer-style embedding (inverted: tokens = variables) ---
        # We keep the repo's embedding; x_mark_enc can be simple sin/cos features.
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, "timeF", "h", dropout)

        # --- Encoder (self-attention over variable tokens) ---
        attn = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.encoder = Encoder(
            [EncoderLayer(attn, d_model, d_ff, dropout=dropout, activation="gelu")
             for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        # --- Static-token fusion (intermediate) ---
        self.static_proj = nn.Linear(static_dim, d_model)

        # --- Classifier head (3 logits) ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim)  # logits for 1h, 1d, 1w
        )

    def _build_time_mark(self, B, L, device):
        # Simple, deterministic time features if you don’t already have them:
        # sin/cos of normalized step index (circadian proxy)
        t = torch.arange(L, device=device).float()
        phase = 2.0 * math.pi * t / L
        sin_t = torch.sin(phase)[None, :, None].repeat(B, 1, 1)
        cos_t = torch.cos(phase)[None, :, None].repeat(B, 1, 1)
        return torch.cat([sin_t, cos_t], dim=-1)  # [B, L, 2]

    def forward(self, static_x, seq_x):
        """
        static_x: [B, static_dim]
        seq_x:    [B, L, N]
        """
        B, L, N = seq_x.shape
        assert L == self.seq_len and N == self.n_vars, f"Expected [{self.seq_len},{self.n_vars}], got [{L},{N}]"

        x_enc = seq_x  # [B, L, N]
        # Optional normalization like original iTransformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # Simple time markers (or pass your own calendar features here)
        if self.add_time_features:
            x_mark_enc = self._build_time_mark(B, L, x_enc.device)  # [B, L, 2]
        else:
            # zero markers if you prefer
            x_mark_enc = torch.zeros(B, L, 2, device=x_enc.device)

        # Inverted embedding → [B, N, d_model]
        enc_tokens = self.enc_embedding(x_enc, x_mark_enc)

        # Append a STATIC token (projected EMR) → intermediate fusion
        static_tok = self.static_proj(static_x).unsqueeze(1)          # [B, 1, d_model]
        tokens = torch.cat([enc_tokens, static_tok], dim=1)           # [B, N+1, d_model]

        # Encoder over tokens
        enc_out, _ = self.encoder(tokens, attn_mask=None)             # [B, N+1, d_model]

        # Use the static token as a CLS-like summary
        cls = enc_out[:, -1, :]                                       # [B, d_model]
        logits = self.classifier(cls)                                  # [B, 3]
        return logits



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
