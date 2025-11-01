import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision import models

# ============================================================
# Model (multilabel MM-IMDb)
# ============================================================
class MMIMDbFusionModel(nn.Module):
    """
    Text: BERT (pooler_output, 768d)
    Image: ResNet-18 (512d) → Linear(image_output_dim)
    Fusion: concat → FC → BN → ReLU → classifier (C logits)
    """
    def __init__(
        self,
        text_hidden_dim: int = 768,
        image_output_dim: int = 256,
        fusion_output_dim: int = 256,
        output_size: int = None,               # if None, inferred from labels.json
        freeze_bert: bool = False,
        pretrained_vision: bool = True,
        bert_name: str = "bert-base-uncased",
    ):
        super().__init__()

        # --- Text encoder ---
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # --- Image encoder (ResNet-18) ---
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_vision else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained_vision)
        resnet.fc = nn.Identity()  # expose 512-d features
        self.image_backbone = resnet
        self.image_proj = nn.Linear(512, image_output_dim)

        # --- Fusion head ---
        fused_dim = text_hidden_dim + image_output_dim
        self.fusion_fc = nn.Linear(fused_dim, fusion_output_dim)
        self.fusion_bn = nn.BatchNorm1d(fusion_output_dim)

        # --- Classifier ---
        if output_size is None:
            import json, os
            labels_json = os.path.abspath("./all_in_one_dataset/labels.json")
            with open(labels_json, "r") as f:
                classes = json.load(f)
            output_size = len(classes)

        self.classifier = nn.Linear(fusion_output_dim, output_size)

    def forward(self, input_ids, attention_mask, image):
        # Text features
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_out.pooler_output  # [B, 768]

        # Image features
        img_feat_512 = self.image_backbone(image)      # [B, 512]
        image_features = self.image_proj(img_feat_512) # [B, image_output_dim]

        # Fuse
        fused = torch.cat([text_features, image_features], dim=1)
        fusion = self.fusion_fc(fused)
        fusion = self.fusion_bn(fusion)
        fusion = F.relu(fusion)

        # Raw logits for BCEWithLogitsLoss (no sigmoid here)
        return self.classifier(fusion)


# ============================================================
# Training / Eval for multilabel (BCE + micro-F1)
# ============================================================
def train_torch(mu: float = 0.0, pos_weight=None):
    """
    Multilabel training:
    - BCEWithLogitsLoss
    - Optional FedProx proximal term
    - Layer-wise LR (BERT lower LR)
    """
    def train(model, dataloader, epochs, cfg=None, *, optimizer=None, device=None, global_params=None):
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if isinstance(pos_weight, torch.Tensor) else None)

        # Keep a snapshot of global params for FedProx
        global_weights = None
        if global_params is not None:
            global_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters()}

        if optimizer is None:
            bert_params, non_bert_params = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                (bert_params if n.startswith("bert.") else non_bert_params).append(p)

            optimizer = torch.optim.AdamW(
                [
                    {"params": bert_params,     "lr": 1e-5, "weight_decay": 0.01},
                    {"params": non_bert_params, "lr": 1e-4, "weight_decay": 0.01},
                ]
            )

        model.train()
        for epoch in range(int(epochs)):
            total_loss, total = 0.0, 0
            tp = fp = fn = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                image = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)  # [B, C] float multi-hot

                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
                loss = criterion(logits, labels)

                # FedProx
                if (mu > 0.0) and (global_weights is not None):
                    prox = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in global_weights:
                            prox = prox + torch.norm(param - global_weights[name]) ** 2
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_loss += float(loss.item()) * labels.size(0)
                    total += int(labels.size(0))

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    tp += int(((preds == 1) & (labels == 1)).sum().item())
                    fp += int(((preds == 1) & (labels == 0)).sum().item())
                    fn += int(((preds == 0) & (labels == 1)).sum().item())

            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = (2 * prec * rec) / max(prec + rec, 1e-8)
            avg_loss = total_loss / max(1, total)
            print(f"🌀 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | micro-F1: {f1:.4f}")

        model.to("cpu")
        return model
    return train


def test_torch():
    def test(model, dataloader, cfg=None, *, device=None):
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = model.to(device).eval()
        criterion = nn.BCEWithLogitsLoss()

        total_loss, total = 0.0, 0
        tp = fp = fn = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                image = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
                total += int(labels.size(0))

                preds = (torch.sigmoid(logits) >= 0.5).float()
                tp += int(((preds == 1) & (labels == 1)).sum().item())
                fp += int(((preds == 1) & (labels == 0)).sum().item())
                fn += int(((preds == 0) & (labels == 1)).sum().item())

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = (2 * prec * rec) / max(prec + rec, 1e-8)
        avg_loss = total_loss / max(1, total)
        model.to("cpu")
        # Return shape compatible with your client/server (metric key 'test_f1_macro')
        return avg_loss, f1, {"test_f1_macro": f1}
    return test
