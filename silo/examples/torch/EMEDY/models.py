# models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# =========================
# 序列模型：LSTM 二分类
# =========================
import torch
import torch.nn as nn

class SleepLSTM(nn.Module):
    """
    x: [B, T, F]  (默认 T=6, F=4)
    输出: logits [B]
    """
    def __init__(
        self,
        input_size: int = 4,                 # 特征数 F
        hidden_sizes=(128, 64, 32),          # 三层 LSTM 宽度
        proj_size: int = 16,                 # 全连接瓶颈
        dropout: float = 0.3,                # FC 之间的 dropout
        bidirectional: bool = False,         # 可选：双向 LSTM
        output_size: int = 1,                # 为兼容上层读取（实际仍输出1个logit）
        **kwargs                               # 吞掉多余配置字段
    ):
        super().__init__()
        self.output_size = output_size

        hs1, hs2, hs3 = hidden_sizes
        d = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hs1, batch_first=True, bidirectional=bidirectional)
        self.do1   = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hs1*d, hidden_size=hs2, batch_first=True, bidirectional=bidirectional)
        self.do2   = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(input_size=hs2*d, hidden_size=hs3, batch_first=True, bidirectional=bidirectional)

        feat_last = hs3 * d
        self.fc1  = nn.Linear(feat_last, proj_size)
        self.act1 = nn.ReLU()
        self.fc2  = nn.Linear(proj_size, 1)  # 二分类：单 logit

        # 可选：把 forget gate 偏置初始化为正数，提升早期记忆能力
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            for names in ["bias_ih_l0", "bias_hh_l0"]:
                if hasattr(lstm, names):
                    b = getattr(lstm, names)
                    # bias 排布为 [i,f,g,o] 四块，各占 hidden 大小
                    hidden = lstm.hidden_size
                    b.data[hidden:2*hidden] += 1.0  # forget gate 偏置 +1
            if bidirectional:
                for names in ["bias_ih_l0_reverse", "bias_hh_l0_reverse"]:
                    if hasattr(lstm, names):
                        b = getattr(lstm, names)
                        hidden = lstm.hidden_size
                        b.data[hidden:2*hidden] += 1.0

    def forward(self, x):                 # x: [B, T, F]
        x, _ = self.lstm1(x)
        x = self.do1(x)
        x, _ = self.lstm2(x)
        x = self.do2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]                   # 取最后时刻
        x = self.act1(self.fc1(x))
        return self.fc2(x).squeeze(-1)    # [B]

# =========================
# 计数与指标工具（micro）
# =========================
@torch.no_grad()
def _counts_from_logits(logits, y_true):
    logits = logits.view(-1, 1)                  # [B,1]
    y_true = y_true.view(-1, 1).long()           # [B,1], {0,1}
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    correct = (preds == y_true).sum().item()
    tp = ((preds == 1) & (y_true == 1)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()
    total = y_true.numel()
    return correct, tp, fp, fn, total


def _estimate_class_weight_from_loader(dl):
    """从 loader 的标签估算 (w0, w1)，用于处理类别不平衡。"""
    pos = 0
    total = 0
    with torch.no_grad():
        for _, y in dl:
            y = y.view(-1)
            pos += int((y > 0.5).sum().item())
            total += int(y.shape[0])
    neg = total - pos
    if pos == 0 or neg == 0:
        return (1.0, 1.0)
    # 令正类权重与负类数量成比例（常见做法）
    return (1.0, neg / pos)


# =========================
# 训练/验证主循环（参考你的实现）
# =========================
def fit_torch_from_loaders(
    model,
    dl_train,
    dl_val=None,
    epochs=55,
    lr=1e-3,
    class_weight=(1.0, 1.0),   # (w0, w1) for {neg,pos}
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 1) 用 pos_weight 实现 class_weight
    w0, w1 = map(float, class_weight)
    pos_weight = torch.tensor([w1 / max(w0, 1e-8)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        # -------- Train --------
        model.train()
        tr_loss_sum = 0.0
        tr_correct = tr_tp = tr_fp = tr_fn = tr_total = 0

        for xb, yb in dl_train:
            xb = xb.to(device)                          # [B,T,F]
            yb = yb.to(device).float().view(-1, 1)      # [B,1]

            logits = model(xb).view(-1, 1)              # [B,1]
            # 若想严格复刻 Keras 的 class_weight 尺度，可乘以 w0（可选）
            loss = criterion(logits, yb) * w0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * yb.size(0)

            c, tp, fp, fn, tot = _counts_from_logits(logits.detach(), yb)
            tr_correct += c; tr_tp += tp; tr_fp += fp; tr_fn += fn; tr_total += tot

        tr_loss = tr_loss_sum / max(tr_total, 1)
        tr_acc  = tr_correct / max(tr_total, 1)
        tr_prec = tr_tp / max(tr_tp + tr_fp, 1) if (tr_tp + tr_fp) > 0 else 0.0
        tr_rec  = tr_tp / max(tr_tp + tr_fn, 1) if (tr_tp + tr_fn) > 0 else 0.0

        # -------- Val（可选） --------
        if dl_val is not None:
            model.eval()
            va_loss_sum = 0.0
            va_correct = va_tp = va_fp = va_fn = va_total = 0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb = xb.to(device)
                    yb = yb.to(device).float().view(-1, 1)

                    logits = model(xb).view(-1, 1)
                    loss = criterion(logits, yb) * w0
                    va_loss_sum += loss.item() * yb.size(0)

                    c, tp, fp, fn, tot = _counts_from_logits(logits, yb)
                    va_correct += c; va_tp += tp; va_fp += fp; va_fn += fn; va_total += tot

            va_loss = va_loss_sum / max(va_total, 1)
            va_acc  = va_correct / max(va_total, 1)
            va_prec = va_tp / max(va_tp + va_fp, 1) if (va_tp + va_fp) > 0 else 0.0
            va_rec  = va_tp / max(va_tp + va_fn, 1) if (va_tp + va_fn) > 0 else 0.0

            print(f"[{epoch:03d}/{epochs}] "
                  f"loss={tr_loss:.4f} acc={tr_acc:.4f} prec={tr_prec:.4f} rec={tr_rec:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_prec={va_prec:.4f} val_rec={va_rec:.4f}")
        else:
            print(f"[{epoch:03d}/{epochs}] "
                  f"loss={tr_loss:.4f} acc={tr_acc:.4f} prec={tr_prec:.4f} rec={tr_rec:.4f}")

    model.to("cpu")
    return model


@torch.no_grad()
def evaluate_loader(model, dl, device=None, class_weight=(1.0, 1.0)):
    # 评测阶段可固定在 CPU（与原实现一致）；如需 GPU，把这里改回自动选择
    device = device or "cpu"
    model.to(device)
    model.eval()

    w0, w1 = map(float, class_weight)
    pos_weight = torch.tensor([w1 / max(w0, 1e-8)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tot_loss = 0.0
    correct = tp = fp = fn = total = 0

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device).float().view(-1, 1)
        logits = model(xb).view(-1, 1)
        loss = criterion(logits, yb) * w0

        tot_loss += loss.item() * yb.size(0)
        c, tpp, fpp, fnn, tot = _counts_from_logits(logits, yb)
        correct += c; tp += tpp; fp += fpp; fn += fnn; total += tot

    acc  = correct / max(total, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    rec  = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    loss = tot_loss / max(total, 1)
    return {"loss": loss, "acc": acc, "prec": prec, "rec": rec}


# =========================
# 对接你项目需要的闭包接口
# =========================
def train_torch():
    """
    返回: custom_train(model, train_loader, epochs, cfg) -> model
    - 自动估算类权重 (w0,w1)
    - 使用 fit_torch_from_loaders 做完整训练
    - 不强依赖 val_loader（如需可另行调用 evaluate_loader）
    """
    def custom_train(model, train_loader, epochs, cfg):
        lr = float(getattr(cfg, "learning_rate", 1e-3))
        # 从训练集估算类权重，缓解不平衡
        class_weight = _estimate_class_weight_from_loader(train_loader)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = fit_torch_from_loaders(
            model=model,
            dl_train=train_loader,
            dl_val=None,                # 如需验证，可在此位置传入 val_loader
            epochs=int(epochs),
            lr=lr,
            class_weight=class_weight,
            device=device
        )
        return model
    return custom_train


def test_torch():
    """
    返回: custom_test(model, test_loader, cfg) -> (loss, num_examples, metrics)
    metrics: {"acc":..., "prec":..., "rec":..., "f1":...}
    """
    def custom_test(model, test_loader, cfg):
        stats = evaluate_loader(model, test_loader, device="cpu", class_weight=(1.0, 1.0))
        loss = float(stats["loss"])
        acc  = float(stats["acc"])
        prec = float(stats["prec"]); rec = float(stats["rec"])
        f1 = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
        # 关键：第二个返回值改为样本数（num_examples）
        num_examples = int(getattr(getattr(test_loader, "dataset", []), "__len__", lambda: 0)())
        metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
        return loss, num_examples, metrics
    return custom_test


