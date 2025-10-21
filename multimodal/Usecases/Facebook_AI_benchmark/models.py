import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision import models


# ============================================================
# Model
# ============================================================
class HatefulMemesFusionModel(nn.Module):
    """
    Text: BERT (pooler_output, 768d)
    Image: ResNet-18 (512d) → Linear(image_output_dim)
    Fusion: concat → FC → BN → ReLU → classifier
    """
    def __init__(
        self,
        text_hidden_dim: int = 768,
        image_output_dim: int = 256,
        fusion_output_dim: int = 256,
        output_size: int = 2,
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
            # Older torchvision fallback
            resnet = models.resnet18(pretrained=pretrained_vision)

        resnet.fc = nn.Identity()  # expose 512-d features
        self.image_backbone = resnet
        self.image_proj = nn.Linear(512, image_output_dim)

        # --- Fusion head ---
        fused_dim = text_hidden_dim + image_output_dim
        self.fusion_fc = nn.Linear(fused_dim, fusion_output_dim)
        self.fusion_bn = nn.BatchNorm1d(fusion_output_dim)
        self.classifier = nn.Linear(fusion_output_dim, output_size)

    def forward(self, input_ids, attention_mask, image):
        # Text
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_out.pooler_output  # [B, 768]

        # Image
        img_feat_512 = self.image_backbone(image)      # [B, 512]
        image_features = self.image_proj(img_feat_512) # [B, image_output_dim]

        # Fuse
        fused = torch.cat([text_features, image_features], dim=1)
        fusion = self.fusion_fc(fused)
        fusion = self.fusion_bn(fusion)
        fusion = F.relu(fusion)

        return self.classifier(fusion)


# ============================================================
# Loss
# ============================================================
class FocalLoss(nn.Module):
    """Optional focal loss for class imbalance (behaves like CE if gamma=0)."""
    def __init__(self, gamma: float = 0.0, weight=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        if self.gamma <= 0:
            loss = ce
        else:
            pt = torch.exp(-ce)
            loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# Training / Eval
# ============================================================
def train_torch(mu: float = 0.0, class_weights=None, focal_gamma: float = 0.0):
    """
    - Optional FocalLoss (set focal_gamma>0)
    - FedProx proximal term over *parameters*
    - Layer-wise LR if optimizer is None (BERT lower LR)
    """
    def train(model, dataloader, epochs, cfg=None, *, optimizer=None, device=None, global_params=None):
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = model.to(device)

        weight = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(gamma=focal_gamma, weight=weight)

        # Keep a snapshot of global params for FedProx
        global_weights = None
        if global_params is not None:
            # Align names with this model's parameters only
            global_weights = {n: p.clone().detach().to(device)
                              for n, p in model.named_parameters()}

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
            total_loss, total_correct, total = 0.0, 0, 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                image = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

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
                    preds = logits.argmax(dim=1)
                    total_correct += int((preds == labels).sum().item())
                    total += int(labels.size(0))

            avg_loss = total_loss / max(1, total)
            acc = total_correct / max(1, total)
            print(f"🌀 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        # Keep API parity with your MNIST trainer
        model.to("cpu")
        return model
    return train


def test_torch(class_weights=None):
    def test(model, dataloader, cfg=None, *, device=None):
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = model.to(device).eval()

        weight = class_weights.to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight)

        total_loss, total_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                image = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)

                preds = logits.argmax(dim=1)
                total_correct += int((preds == labels).sum().item())
                total += int(labels.size(0))

        avg_loss = total_loss / max(1, total)
        acc = total_correct / max(1, total)
        model.to("cpu")
        return avg_loss, acc, {"accuracy": acc}
    return test
