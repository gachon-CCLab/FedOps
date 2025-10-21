# flclient_patch.py
from fedops.client.client_fl import FLClient as BaseClien
import torch

class MyFLClient(BaseClient):
    def fit(self, parameters, config):
        # 1) load global weights
        self.set_parameters(parameters)

        # 2) pre-eval on validation to get p_k before (acc_before)
        _, acc_before, _ = self.test_torch(self.model, self.val_loader, self.cfg)

        # 3) call original training path by delegating to BaseClient.fit
        params_prime, num_examples_train, base_metrics = super().fit(parameters, config)

        # 4) post-eval on validation (p_k after = val_accuracy)
        val_loss, acc_after, _ = self.test_torch(self.model, self.val_loader, self.cfg)

        # 5) extra metrics your strategy expects
        # grad_norm best-effort (may be 0.0 if grads cleared by trainer)
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += float(torch.norm(g, p=2).item() ** 2)
        grad_norm = (total_sq ** 0.5) if total_sq > 0.0 else 0.0

        # class-imbalance (try to infer; fallback to 0.0)
        try:
            pos = 0
            total = 0
            ds = self.train_loader.dataset
            for i in range(len(ds)):
                lbl = int(ds[i]["label"] if isinstance(ds[i], dict) else ds[i][1])
                pos += (lbl == 1)
                total += 1
            imbalance_ratio = (pos / total) if total else 0.0
        except Exception:
            imbalance_ratio = 0.0

        # modality flags (if you use them)
        m_img = 1
        m_txt = 1

        enriched = dict(base_metrics)  # includes train_/val_ keys from BaseClient
        enriched.update({
            "c_k": float(acc_after - acc_before),
            "p_k": float(acc_after),
            "val_loss": float(val_loss),
            "grad_norm": float(grad_norm),
            "imbalance_ratio": float(imbalance_ratio),
            "m_img": int(m_img),
            "m_txt": int(m_txt),
        })

        # make sure num_examples is dataset size (n_k)
        n_k = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else num_examples_train
        return params_prime, n_k, enriched
