# PatchTST.py (or any file you prefer)
import torch
from torch import nn
from layers.PatchTST_backbone import PatchTST_backbone

class PatchTSTClassifier(nn.Module):
    """
    Multi-label classifier for 3 horizons.
    Inputs:
      static_x: [B, 14]
      seq_x   : [B, 192, 25]
    Output:
      logits  : [B, 3]
    """
    def __init__(
        self,
        seq_len=192, n_vars=25, static_dim=14, out_dim=3,
        d_model=128, n_heads=4, e_layers=4, d_ff=512,
        patch_len=16, stride=8, dropout=0.1, head_dropout=0.1,
        revin=True, affine=True, subtract_last=False,
        padding_patch="end", pe="zeros", learn_pe=True,
        individual=False, attn_dropout=0.0, norm="BatchNorm",
        res_attention=True, pre_norm=False, use_norm=True,
    ):
        super().__init__()
        self.output_dim = out_dim

        # We set target_window=1; we’ll request encoder features via return_features=True
        self.backbone = PatchTST_backbone(
            c_in=n_vars, context_window=seq_len, target_window=1,
            patch_len=patch_len, stride=stride, max_seq_len=1024,
            n_layers=e_layers, d_model=d_model, n_heads=n_heads,
            d_k=None, d_v=None, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act="gelu",
            key_padding_mask='auto', padding_var=None, attn_mask=None,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=False,
            pe=pe, learn_pe=learn_pe, fc_dropout=dropout, head_dropout=head_dropout,
            padding_patch=padding_patch, pretrain_head=False, head_type='flatten',
            individual=individual, revin=revin, affine=affine, subtract_last=subtract_last,
            verbose=False
        )

        self.static_proj = (nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model) if use_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout),
        ) if static_dim and static_dim > 0 else None)

        cls_in = d_model * (2 if self.static_proj is not None else 1)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, d_model), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(d_model, out_dim),
        )

    @torch.no_grad()
    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, C, D, P]  → mean over patches P, then variables C → [B, D]
        return feats.mean(dim=-1).mean(dim=1)

    def forward(self, static_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        # seq_x: [B, L, C] → backbone wants [B, C, L]
        x_cf = seq_x.permute(0, 2, 1)
        feats = self.backbone(x_cf, return_features=True)   # [B, C, D, P], RevIN-normalized
        h = self._pool(feats)                               # [B, D]
        if self.static_proj is not None:
            h = torch.cat([h, self.static_proj(static_x)], dim=-1)
        return self.classifier(h)                           # [B, 3] logits

    
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
import json
def load_hospital_mapping():
    with open("hospital_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)
