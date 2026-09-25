import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(counts, mode, beta=0.999):
    counts = torch.tensor(counts, dtype=torch.float).clamp(min=1)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "sqrt_inverse":
        w = 1.0 / counts.sqrt()
    elif mode == "effective":
        # Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples"
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    else:
        raise ValueError(f"class_weight không hợp lệ: {mode}")
    return w / w.sum() * len(counts)  # chuẩn hóa để trung bình weight = 1


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (Lin et al. 2017).

    (1 - p_t)^gamma giảm trọng số các mẫu đã phân loại dễ (p_t cao), để gradient tập trung
    vào mẫu khó — thường là mẫu của lớp hiếm. alpha là class weight (có thể None)."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - log_pt.exp()) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def build_loss(loss_cfg, class_counts):
    mode = loss_cfg.get("class_weight")
    weight = compute_class_weights(class_counts, mode, loss_cfg.get("beta", 0.999)) if mode else None
    if loss_cfg.name == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=loss_cfg.get("label_smoothing", 0.0))
    if loss_cfg.name == "focal":
        return FocalLoss(gamma=loss_cfg.get("gamma", 2.0), alpha=weight)
    raise ValueError(f"loss không hợp lệ: {loss_cfg.name}")
