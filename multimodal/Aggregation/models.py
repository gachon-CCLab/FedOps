import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision import models


# ---------- Model ----------
class HatefulMemesFusionModel(nn.Module):
    """
    Text: BERT (pooler_output, 768d)
    Image: ResNet-18 backbone (512d) -> Linear to image_output_dim
    Fusion: [text; image] -> BN -> ReLU -> classifier
    """
    def __init__(
        self,
        text_hidden_dim: int = 768,
        image_output_dim: int = 256,          # bump default a bit
        fusion_output_dim: int = 256,
        output_size: int = 2,
        freeze_bert: bool = False,
        pretrained_vision: bool = True,
    ):
        super().__init__()

        # --- Text encoder ---
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # --- Image encoder (ResNet-18) ---
        # Output after global pooling is 512-d
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_vision else None)
        resnet.fc = nn.Identity()  # keep 512-d features
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

        # Image: expect normalized, 3xHxW
        img_feat_512 = self.image_backbone(image)         # [B, 512]
        image_features = self.image_proj(img_feat_512)    # [B, image_output_dim]

        # Fuse
        fused = torch.cat([text_features, image_features], dim=1)
        fusion = self.fusion_fc(fused)
        fusion = self.fusion_bn(fusion)
        fusion = F.relu(fusion)

        return self.classifier(fusion)


# ---------- Loss helpers ----------
class FocalLoss(nn.Module):
    """Optional focal loss for class imbalance (works like weighted CE if gamma=0)."""
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


# ---------- Training & Testing ----------
def train_torch(mu: float = 0.0, class_weights=None, focal_gamma: float = 0.0):
    """
    Returns a training fn with:
      - Optional FocalLoss (set focal_gamma>0, still supports class weights)
      - FedProx with proper state_dict key matching
      - Layer-wise LR if optimizer is None (BERT lower LR, fusion/vision higher LR)
    """

    def train(model, dataloader, optimizer=None, epochs=1, cfg=None, global_params=None):
        device = next(model.parameters()).device

        weight = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(gamma=focal_gamma, weight=weight)

        model.train()

        # --- Keep a consistent copy of global weights for FedProx (match named_parameters keys) ---
        global_weights = None
        if global_params is not None:
            # Ensure keys match param names (not buffers)
            # Save a snapshot from *this* model so names align
            global_weights = {k: v.clone().detach().to(device)
                              for k, v in model.state_dict().items()
                              if k in dict(model.named_parameters())}

        # --- If no optimizer given, build one with layer-wise LR ---
        if optimizer is None:
            # Lower LR for BERT, higher for vision + fusion + classifier
            bert_params = []
            non_bert_params = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("bert."):
                    bert_params.append(p)
                else:
                    non_bert_params.append(p)

            optimizer = torch.optim.AdamW(
                [
                    {"params": bert_params, "lr": 1e-5, "weight_decay": 0.01},
                    {"params": non_bert_params, "lr": 1e-4, "weight_decay": 0.01},
                ]
            )

        for epoch in range(epochs):
            total_loss, total_correct, total = 0.0, 0, 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                image = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
                loss = criterion(logits, labels)

                # --- FedProx proximal term (only over params, not buffers) ---
                if (mu > 0.0) and (global_weights is not None):
                    prox = 0.0
                    for name, param in model.named_parameters():
                        if (param.requires_grad) and (name in global_weights):
                            prox += torch.norm(param - global_weights[name]) ** 2
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_loss = total_loss / max(1, total)
            accuracy = total_correct / max(1, total)
            print(f"🌀 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

        return avg_loss, accuracy, {"train_accuracy": accuracy}

    return train


def test_torch(class_weights=None):
    """Vanilla eval with optional class weights (use focal in training only to keep eval stable)."""
    def test(model, dataloader, cfg=None):
        device = next(model.parameters()).device
        weight = class_weights.to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight)

        model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                image = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        accuracy = total_correct / max(1, total)
        return avg_loss, accuracy, {"test_accuracy": accuracy}

    return test
