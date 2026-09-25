import torch.nn as nn

from vimmsd.models.fusion import FUSIONS
from vimmsd.models.image_encoder import ImageEncoder
from vimmsd.models.text_encoder import TextEncoder

MODALITY_NEEDS = {
    "text": (True, False),
    "image": (False, True),
    "concat": (True, True),
    "cross_attn": (True, True),
}


class SarcasmClassifier(nn.Module):
    """Lắp ghép encoder + fusion + head MLP phân loại.

    Tách `encode` và `classify` để phần analysis (Grad-CAM, tráo ảnh) có thể can thiệp
    vào feature ở giữa mà không cần chạy lại encoder."""

    def __init__(self, modality, text_encoder, image_encoder, fusion_cfg, num_classes, head_hidden, dropout):
        super().__init__()
        self.modality = modality
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = FUSIONS[modality](
            text_dim=text_encoder.hidden_size if text_encoder else None,
            image_dim=image_encoder.hidden_size if image_encoder else None,
            dropout=dropout,
            **fusion_cfg,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fusion.out_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer if self.text_encoder else None

    @property
    def image_processor(self):
        return self.image_encoder.image_processor if self.image_encoder else None

    def encode(self, batch):
        text = image = None
        if self.text_encoder is not None:
            text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        if self.image_encoder is not None:
            image = self.image_encoder(batch["pixel_values"])
        return text, image

    def classify(self, text, image):
        return self.head(self.fusion(text, image))

    def forward(self, batch):
        return self.classify(*self.encode(batch))

    def param_groups(self, lr_encoder, lr_head, weight_decay):
        encoder_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_encoder = name.startswith(("text_encoder.", "image_encoder."))
            (encoder_params if is_encoder else head_params).append(p)
        groups = [{"params": head_params, "lr": lr_head, "weight_decay": weight_decay}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay})
        return groups


def build_model(cfg):
    m = cfg.model
    if m.modality not in MODALITY_NEEDS:
        raise ValueError(f"model.modality phải thuộc {list(MODALITY_NEEDS)}, nhận được {m.modality!r}")
    needs_text, needs_image = MODALITY_NEEDS[m.modality]
    text_encoder = TextEncoder(m.text_encoder.name, m.text_encoder.freeze) if needs_text else None
    image_encoder = ImageEncoder(m.image_encoder.name, m.image_encoder.freeze) if needs_image else None
    return SarcasmClassifier(
        modality=m.modality,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_cfg=dict(m.get("fusion", {})),
        num_classes=len(cfg.data.labels),
        head_hidden=m.head_hidden,
        dropout=m.dropout,
    )
