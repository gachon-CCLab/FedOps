# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class HatefulMemesFusionModel(nn.Module):
    def __init__(
        self,
        text_hidden_dim=768,
        image_output_dim=128,
        fusion_output_dim=256,
        output_size=2,
    ):
        super().__init__()
        # Text Encoder: BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Image Encoder: simple CNN
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.flatten = nn.Flatten()
        self.image_fc = nn.Linear(64 * 4 * 4, image_output_dim)
        # Fusion layer + classifier
        self.fusion_fc = nn.Linear(text_hidden_dim + image_output_dim, fusion_output_dim)
        self.classifier = nn.Linear(fusion_output_dim, output_size)

    def forward(self, input_ids, attention_mask, image):
        # Text features
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = bert_out.pooler_output  # (batch_size, 768)
        # Image features
        img_feat = self.image_encoder(image)
        img_feat = self.flatten(img_feat)
        img_feat = self.image_fc(img_feat)  # (batch_size, image_output_dim)
        # Fusion
        fused = torch.cat((txt_feat, img_feat), dim=1)
        fused = F.relu(self.fusion_fc(fused))
        return self.classifier(fused)


def train_torch():
    def train(model, train_loader, epochs, cfg):
        device = next(model.parameters()).device
        model.to(device)

        # 1) Pre-train evaluation on val set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in cfg.val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                preds = out.argmax(dim=1)
                correct += (preds == batch["label"].to(device)).sum().item()
                total += len(preds)
        p_before = correct / total if total > 0 else 0.0

        # 2) Local training
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for _ in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                loss = criterion(out, batch["label"].to(device))
                loss.backward()
                optimizer.step()

        # 3) Post-train evaluation on val set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in cfg.val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                preds = out.argmax(dim=1)
                correct += (preds == batch["label"].to(device)).sum().item()
                total += len(preds)
        p_after = correct / total if total > 0 else 0.0

        # Store for test_torch to access
        model.p_before = p_before
        model.p_after = p_after

        return model

    return train


def test_torch():
    def test(model, test_loader, cfg):
        device = next(model.parameters()).device
        model.to(device).eval()

        # Standard test loss & accuracy
        criterion = nn.CrossEntropyLoss()
        total_loss, total_correct, total_examples = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                loss = criterion(out, batch["label"].to(device))
                total_loss += loss.item()
                preds = out.argmax(dim=1)
                total_correct += (preds == batch["label"].to(device)).sum().item()
                total_examples += len(preds)

        avg_loss = total_loss / total_examples if total_examples else 0.0
        accuracy = total_correct / total_examples if total_examples else 0.0

        # Build summary payload
        p_before = getattr(model, "p_before", accuracy)
        p_after = getattr(model, "p_after", accuracy)
        contrib = p_after - p_before
        # Coverage flags (text & image only)
        cov_img, cov_txt = 1.0, 1.0

        metrics = {
            "loss":    avg_loss,
            "accuracy": accuracy,
            "perf":    p_after,
            "contrib": contrib,
            "cov_img": cov_img,
            "cov_txt": cov_txt,
        }
        return avg_loss, accuracy, metrics

    return test
