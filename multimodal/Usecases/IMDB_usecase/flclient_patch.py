from fedops.client.client_fl import FLClient as BaseClient
import torch

class MyFLClient(BaseClient):
    def fit(self, parameters, config):
        # 1) load global weights
        self.set_parameters(parameters)

        # 2) pre-eval on validation (now returns (loss, f1, metrics))
        _, f1_before, _ = self.test_torch(self.model, self.val_loader, self.cfg)

        # 3) delegate to BaseClient.fit() to run training
        params_prime, num_examples_train, base_metrics = super().fit(parameters, config)

        # 4) post-eval on validation
        val_loss, f1_after, _ = self.test_torch(self.model, self.val_loader, self.cfg)

        # 5) extra metrics needed by the strategy
        # grad_norm best-effort
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += float(torch.norm(g, p=2).item() ** 2)
        grad_norm = (total_sq ** 0.5) if total_sq > 0.0 else 0.0

        # multilabel imbalance: mean positive rate across labels
        try:
            pos_sum = 0.0
            count = 0
            ds = self.train_loader.dataset
            for i in range(len(ds)):
                y = ds[i]["label"].float().view(-1)  # multi-hot
                pos_sum += float(y.mean().item())
                count += 1
            imbalance_ratio = (pos_sum / max(count, 1))
        except Exception:
            imbalance_ratio = 0.0

        # modality flags (present in your pipeline; keep 1/1 for now)
        m_img = 1
        m_txt = 1

        enriched = dict(base_metrics)  # may include train_/val_ keys from BaseClient
        enriched.update({
            "c_k": float(f1_after - f1_before),  # change in validation F1
            "p_k": float(f1_after),              # post-train validation F1
            "val_loss": float(val_loss),
            "grad_norm": float(grad_norm),
            "imbalance_ratio": float(imbalance_ratio),
            "m_img": int(m_img),
            "m_txt": int(m_txt),
        })

        # make sure num_examples is dataset size (n_k)
        n_k = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else num_examples_train
        return params_prime, n_k, enriched
