import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel


class ImageEncoder(nn.Module):
    """Wrapper CLIP vision (mặc định) hoặc ViT: trả về embedding từng patch và vector pooled."""

    def __init__(self, name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        model_type = AutoConfig.from_pretrained(name).model_type
        self.is_clip = model_type in {"clip", "clip_vision_model"}
        # AutoModel với checkpoint CLIP sẽ load cả text tower, chỉ cần vision tower
        self.model = CLIPVisionModel.from_pretrained(name) if self.is_clip else AutoModel.from_pretrained(name)
        self.image_processor = AutoImageProcessor.from_pretrained(name)
        self.hidden_size = self.model.config.hidden_size
        self.frozen = freeze
        if freeze:
            self.model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state  # (B, 1 + num_patches, H), token 0 là CLS
        pooled = out.pooler_output if self.is_clip else tokens[:, 0]
        return {"tokens": tokens, "pooled": pooled}
