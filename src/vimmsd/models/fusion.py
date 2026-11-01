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


class TextAuxFusion(nn.Module):
    """Caption là feature chính, hệ số cố định 1.

    OCR đứng thứ hai, mô tả VLM đứng thứ ba. Hai hệ số phụ cố định, không học,
    để lần chạy thực sự dùng hai nguồn này thay vì để gate kẹt ở 0.1.
    h = h_caption + ocr_weight * h_ocr + description_weight * h_vlm.
    Mẫu không có OCR hoặc không có mô tả thì feature phụ đó bằng 0.
    """

    def __init__(self, text_dim, ocr_weight=0.5, description_weight=0.25, **_):
        super().__init__()
        self.out_dim = text_dim
        self.ocr_weight = float(ocr_weight)
        self.description_weight = float(description_weight)
        if not 0 <= self.description_weight <= self.ocr_weight <= 1:
            raise ValueError("cần 0 <= description_weight <= ocr_weight <= 1, caption giữ hệ số 1")

    def weights(self):
        return {"caption": 1.0, "ocr": self.ocr_weight, "description": self.description_weight}

    def forward(self, text, image=None):
        hidden = text["pooled"]
        if text.get("ocr") is not None:
            present = text["ocr_present"].unsqueeze(-1).to(hidden.dtype)
            hidden = hidden + self.ocr_weight * text["ocr"] * present
        if text.get("description") is not None:
            present = text["description_present"].unsqueeze(-1).to(hidden.dtype)
            hidden = hidden + self.description_weight * text["description"] * present
        return hidden


class ImageOnlyFusion(nn.Module):
    def __init__(self, image_dim, **_):
        super().__init__()
        self.out_dim = image_dim

    def forward(self, text=None, image=None):
        return image["pooled"]


class ConcatFusion(nn.Module):
    """[t; i] với t = PhoBERT <s>, i = CLIP pooled. Mặc định nối thẳng như Phase 4.

    Các tuỳ chọn chống việc head học thuộc tập train qua vector ảnh cố định (thấy ở fusion_concat_frozen:
    train loss giảm nhanh hơn text-only, early stop sớm, lớp hiếm tụt):
    - `proj_dim`: LayerNorm + Linear(proj_dim) riêng mỗi nhánh trước khi nối, để 2 vector cùng thang đo
      (<s> của PhoBERT không qua LayerNorm cuối, pooled của CLIP đã qua `post_layernorm`).
    - `image_feature_dropout`: dropout thường trên vector ảnh, chỉ khi train.
    - `image_drop_prob` / `text_drop_prob`: modality dropout, mỗi mẫu train bị bỏ hẳn vector ảnh / text
      (thay bằng 0) với xác suất tương ứng; không bao giờ bỏ cả hai trong cùng một mẫu (giữ text).
      Bỏ ảnh buộc head phải phân loại được chỉ từ text, nên phần text không bị "loãng"."""

    def __init__(self, text_dim, image_dim, proj_dim=None, image_feature_dropout=0.0,
                 image_drop_prob=0.0, text_drop_prob=0.0, **_):
        super().__init__()
        if not (0 <= image_drop_prob < 1 and 0 <= text_drop_prob < 1):
            raise ValueError("image_drop_prob và text_drop_prob phải trong [0, 1)")
        self.image_drop_prob = float(image_drop_prob)
        self.text_drop_prob = float(text_drop_prob)
        self.image_feature_dropout = nn.Dropout(image_feature_dropout)
        if proj_dim:
            self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, proj_dim))
            self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, proj_dim))
            self.out_dim = 2 * proj_dim
        else:
            self.text_proj = self.image_proj = nn.Identity()
            self.out_dim = text_dim + image_dim

    def forward(self, text, image):
        t = self.text_proj(text["pooled"])
        i = self.image_proj(self.image_feature_dropout(image["pooled"]))
        if self.training and (self.image_drop_prob or self.text_drop_prob):
            B = t.shape[0]
            drop_image = torch.rand(B, device=t.device) < self.image_drop_prob
            drop_text = (torch.rand(B, device=t.device) < self.text_drop_prob) & ~drop_image
            i = i * (~drop_image).unsqueeze(-1).to(i.dtype)
            t = t * (~drop_text).unsqueeze(-1).to(t.dtype)
        return torch.cat([t, i], dim=-1)


class MultiStreamFusion(nn.Module):
    """Nhiều luồng giữ riêng, nối vector: caption, chữ trong ảnh (OCR), mô tả ảnh (VLM), và vector CLIP nếu có.

    Khác `TextAuxFusion` (cộng các luồng với hệ số đặt tay) và khác cách ghép chuỗi (OCR, mô tả thành segment 2
    của caption): mỗi luồng qua LayerNorm + Linear(proj_dim) riêng rồi nối lại, head tự học trọng số, và vẫn biết
    tín hiệu đến từ caption hay từ ảnh, đúng với cách chia nhãn text-sarcasm / image-sarcasm.

    - Luồng chữ từ ảnh bị thiếu (ảnh không có chữ, không có mô tả) thành vector 0, kèm cờ có/không ở cuối vector,
      để head không nhầm "không có chữ" với "chữ trung tính".
    - `interactions`: thêm t*s và |t - s| giữa caption t và từng luồng chữ từ ảnh s, đo mức khớp/lệch giữa
      caption và nội dung ảnh (sarcasm dạng nói một đằng, ảnh một nẻo).
    - `aux_drop_prob`: modality dropout, mỗi mẫu train bỏ từng luồng từ ảnh (OCR, mô tả, CLIP) với xác suất này,
      coi như luồng đó vắng mặt; buộc head dùng được caption khi thiếu ảnh. Caption không bao giờ bị bỏ.
    - `image_feature_dropout`: dropout trên vector CLIP (chống học thuộc qua vector ảnh cố định, xem ConcatFusion)."""

    AUX = ("ocr", "description")

    def __init__(self, text_dim, image_dim=None, proj_dim=256, streams=AUX, interactions=True,
                 aux_drop_prob=0.0, image_feature_dropout=0.0, **_):
        super().__init__()
        streams = tuple(streams)
        if not streams or set(streams) - set(self.AUX):
            raise ValueError(f"streams phải là tập con khác rỗng của {self.AUX}, nhận được {streams}")
        if not 0 <= aux_drop_prob < 1:
            raise ValueError("aux_drop_prob phải trong [0, 1)")
        self.streams = streams
        self.interactions = bool(interactions)
        self.aux_drop_prob = float(aux_drop_prob)
        self.use_image = image_dim is not None

        def proj(dim):
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj_dim))

        self.caption_proj = proj(text_dim)
        self.aux_proj = nn.ModuleDict({name: proj(text_dim) for name in streams})
        if self.use_image:
            self.image_proj = proj(image_dim)
            self.image_feature_dropout = nn.Dropout(image_feature_dropout)
        n_streams = 1 + len(streams) + int(self.use_image)
        self.out_dim = n_streams * proj_dim + len(streams) + (2 * proj_dim * len(streams) if self.interactions else 0)

    def _keep(self, present):
        """Cờ có mặt (B,) sau modality dropout: luồng bị bỏ được coi như vắng mặt."""
        if self.training and self.aux_drop_prob:
            present = present * (torch.rand_like(present) >= self.aux_drop_prob).to(present.dtype)
        return present

    def forward(self, text, image=None):
        t = self.caption_proj(text["pooled"])
        feats, flags, inter = [t], [], []
        for name in self.streams:
            if text.get(name) is None:
                raise ValueError(f"thiếu luồng '{name}': bật data.image_text.use_{name} và dùng file image_preprocessed")
            present = self._keep(text[f"{name}_present"].to(t.dtype))
            s = self.aux_proj[name](text[name]) * present.unsqueeze(-1)
            feats.append(s)
            flags.append(present.unsqueeze(-1))
            if self.interactions:
                # luồng vắng mặt: s = 0 nên t*s = 0; |t - s| cũng phải về 0 để không lặp lại t
                inter += [t * s, (t - s).abs() * present.unsqueeze(-1)]
        if self.use_image:
            i = self.image_proj(self.image_feature_dropout(image["pooled"]))
            feats.append(i * self._keep(torch.ones_like(i[:, 0])).unsqueeze(-1))
        return torch.cat(feats + inter + flags, dim=-1)


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
    """Cross-attention giữa token text và patch ảnh, chọn chiều bằng `direction`:

    - text_to_image: token text là query, attend lên patch ảnh (ảnh chỉ qua phép chiếu).
    - image_to_text: patch ảnh là query, attend lên token text (text chỉ qua phép chiếu).
    - both (mặc định): chạy cả hai chiều trong cùng một lượt.

    Vector đầu ra = [t; i; t*i; |t-i|] với t, i là biểu diễn đã pool của mỗi modality, nên out_dim
    như nhau ở cả ba chế độ và các chế độ so sánh được với nhau. Hai thành phần t*i và |t-i| biểu
    diễn trực tiếp mức độ khớp/lệch giữa ảnh và text, là tín hiệu quan trọng cho sarcasm dạng
    mâu thuẫn giữa 2 modality.

    Attention weight của layer cuối lưu ở `self.last_attn` để vẽ attention map (Tuần 6); chiều không
    chạy thì giá trị tương ứng là None.
    """

    DIRECTIONS = ("both", "text_to_image", "image_to_text")

    def __init__(self, text_dim, image_dim, hidden_dim=512, num_heads=8, num_layers=1, dropout=0.1,
                 direction="both", **_):
        super().__init__()
        if direction not in self.DIRECTIONS:
            raise ValueError(f"direction không hợp lệ: {direction!r}, chọn một trong {self.DIRECTIONS}")
        self.direction = direction
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        def blocks(active):
            n = num_layers if active else 0
            return nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(n)])

        self.text_blocks = blocks(direction in ("both", "text_to_image"))
        self.image_blocks = blocks(direction in ("both", "image_to_text"))
        self.out_dim = hidden_dim * 4
        self.last_attn = None

    def forward(self, text, image):
        t = self.text_proj(text["tokens"])
        i = self.image_proj(image["tokens"])
        text_mask = text["mask"]

        t2i = i2t = None
        for layer in range(max(len(self.text_blocks), len(self.image_blocks))):
            t_new, i_new = t, i
            if self.text_blocks:
                t_new, t2i = self.text_blocks[layer](t, i)
            if self.image_blocks:
                i_new, i2t = self.image_blocks[layer](i, t, context_pad_mask=~text_mask)
            t, i = t_new, i_new
        self.last_attn = {
            "text_to_image": None if t2i is None else t2i.detach(),
            "image_to_text": None if i2t is None else i2t.detach(),
        }

        m = text_mask.unsqueeze(-1).to(t.dtype)
        t_pooled = (t * m).sum(1) / m.sum(1).clamp(min=1.0)
        i_pooled = i.mean(1)
        return torch.cat([t_pooled, i_pooled, t_pooled * i_pooled, (t_pooled - i_pooled).abs()], dim=-1)


FUSIONS = {
    "text": TextOnlyFusion,
    "text_aux": TextAuxFusion,
    "image": ImageOnlyFusion,
    "concat": ConcatFusion,
    "cross_attn": CrossAttentionFusion,
    "multistream": MultiStreamFusion,       # caption + OCR + mô tả VLM, không dùng CLIP
    "multistream_clip": MultiStreamFusion,  # như trên + vector CLIP
}
