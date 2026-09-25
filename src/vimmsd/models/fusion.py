import torch
import torch.nn as nn

# Mọi fusion có cùng interface: forward(text, image) -> (B, out_dim), với
#   text  = {"tokens": (B, Lt, Ht), "mask": (B, Lt) bool, "pooled": (B, Ht)} hoặc None
#   image = {"tokens": (B, Li, Hi), "pooled": (B, Hi)} hoặc None
# để đổi fusion chỉ cần sửa `model.modality` trong config.


class TextOnlyFusion(nn.Module):
    def __init__(self, text_dim, **_):
        super().__init__()
        self.out_dim = text_dim

    def forward(self, text, image=None):
        return text["pooled"]


class ImageOnlyFusion(nn.Module):
    def __init__(self, image_dim, **_):
        super().__init__()
        self.out_dim = image_dim

    def forward(self, text=None, image=None):
        return image["pooled"]


class ConcatFusion(nn.Module):
    def __init__(self, text_dim, image_dim, **_):
        super().__init__()
        self.out_dim = text_dim + image_dim

    def forward(self, text, image):
        return torch.cat([text["pooled"], image["pooled"]], dim=-1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_pad_mask=None):
        attended, weights = self.attn(
            query, context, context, key_padding_mask=context_pad_mask,
            need_weights=True, average_attn_weights=True,
        )
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, weights


class CrossAttentionFusion(nn.Module):
    """Cross-attention 2 chiều: token text attend lên patch ảnh và ngược lại.

    Vector đầu ra = [t; i; t*i; |t-i|] với t, i là biểu diễn đã pool của mỗi modality.
    Hai thành phần t*i và |t-i| biểu diễn trực tiếp mức độ khớp/lệch giữa ảnh và text,
    là tín hiệu quan trọng cho sarcasm dạng mâu thuẫn giữa 2 modality.

    Attention weight của layer cuối lưu ở `self.last_attn` để vẽ attention map (Tuần 6).
    """

    def __init__(self, text_dim, image_dim, hidden_dim=512, num_heads=8, num_layers=1, dropout=0.1, **_):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.image_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_dim = hidden_dim * 4
        self.last_attn = None

    def forward(self, text, image):
        t = self.text_proj(text["tokens"])
        i = self.image_proj(image["tokens"])
        text_mask = text["mask"]

        for t_block, i_block in zip(self.text_blocks, self.image_blocks):
            t_new, t2i = t_block(t, i)
            i_new, i2t = i_block(i, t, context_pad_mask=~text_mask)
            t, i = t_new, i_new
        self.last_attn = {"text_to_image": t2i.detach(), "image_to_text": i2t.detach()}

        m = text_mask.unsqueeze(-1).to(t.dtype)
        t_pooled = (t * m).sum(1) / m.sum(1).clamp(min=1.0)
        i_pooled = i.mean(1)
        return torch.cat([t_pooled, i_pooled, t_pooled * i_pooled, (t_pooled - i_pooled).abs()], dim=-1)


FUSIONS = {
    "text": TextOnlyFusion,
    "image": ImageOnlyFusion,
    "concat": ConcatFusion,
    "cross_attn": CrossAttentionFusion,
}
