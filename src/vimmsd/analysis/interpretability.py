import html
import math

import numpy as np
import torch
from PIL import ImageOps

from vimmsd.models.fusion import CrossAttentionFusion


def _grid(scores):
    """(B, num_patches) -> (B, g, g)."""
    g = int(math.isqrt(scores.shape[-1]))
    return scores[..., : g * g].reshape(*scores.shape[:-1], g, g)


@torch.no_grad()
def cross_attention_maps(model, batch):
    """Lấy attention của layer cross-attention cuối.

    - image_heatmap: trung bình attention của các token text (không tính padding) lên từng patch ảnh.
    - token_scores: trung bình attention của các patch ảnh lên từng token text."""
    if not isinstance(model.fusion, CrossAttentionFusion):
        raise TypeError("attention map chỉ có với model.modality = cross_attn")
    model.eval()
    logits = model(batch)
    attn = model.fusion.last_attn
    mask = batch["attention_mask"].to(attn["text_to_image"].dtype)

    t2i = (attn["text_to_image"] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    image_heatmap = _grid(t2i[:, 1:])  # bỏ token CLS của ảnh
    token_scores = attn["image_to_text"].mean(1) * mask
    return {"logits": logits, "image_heatmap": image_heatmap.cpu(), "token_scores": token_scores.cpu()}


def _last_vision_layer(image_encoder):
    # CLIP: (vision_model.)encoder.layers, ViT: encoder.layer — vị trí khác nhau giữa các bản transformers
    for name, module in image_encoder.model.named_modules():
        if name.endswith(("encoder.layers", "encoder.layer")) and isinstance(module, torch.nn.ModuleList):
            return module[-1]
    raise TypeError("không xác định được layer cuối của image encoder")


def gradcam(model, batch, target_class=None):
    """Grad-CAM trên token patch ở đầu vào block cuối của image encoder.

    cam = ReLU( sum_c  mean_patch(dy/dA_c) * A_c ), A là activation các patch. Dùng được cho mọi
    fusion có ảnh (kể cả concat), và cả khi encoder bị đóng băng: bật requires_grad cho pixel_values
    để autograd vẫn dựng graph qua encoder."""
    if model.image_encoder is None:
        raise TypeError("model không có image encoder")
    model.eval()
    store = {}

    # lấy activation ở ĐẦU VÀO block cuối: output của block cuối chỉ ảnh hưởng tới CLS đi qua
    # chính nó, nên với fusion chỉ dùng CLS (concat) gradient trên các patch ở đó luôn bằng 0
    def hook(_, args, kwargs):
        act = args[0] if args else kwargs["hidden_states"]
        act.retain_grad()
        store["act"] = act

    handle = _last_vision_layer(model.image_encoder).register_forward_pre_hook(hook, with_kwargs=True)
    try:
        batch = dict(batch)
        batch["pixel_values"] = batch["pixel_values"].clone().requires_grad_(True)
        logits = model(batch)
        target = logits.argmax(-1) if target_class is None else torch.full_like(logits[:, 0], target_class).long()
        model.zero_grad(set_to_none=True)
        logits.gather(1, target.unsqueeze(1)).sum().backward()
        act, grad = store["act"][:, 1:], store["act"].grad[:, 1:]
        weights = grad.mean(1, keepdim=True)
        cam = torch.relu((weights * act).sum(-1))
        cam = cam / cam.amax(-1, keepdim=True).clamp(min=1e-8)
    finally:
        handle.remove()
    return {"logits": logits.detach(), "target": target, "heatmap": _grid(cam.detach()).cpu()}


def encoder_view(image, size=224):
    """Ảnh đúng như encoder nhìn thấy (resize cạnh ngắn + center crop kiểu CLIP), để overlay heatmap khớp vị trí."""
    return ImageOps.fit(image.convert("RGB"), (size, size), centering=(0.5, 0.5))


def overlay_heatmap(image, heatmap, ax=None, alpha=0.5, title=None):
    import matplotlib.pyplot as plt

    view = encoder_view(image)
    heat = torch.as_tensor(heatmap, dtype=torch.float)[None, None]
    heat = torch.nn.functional.interpolate(heat, size=view.size[::-1], mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min()).clamp(min=1e-8)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(view)
    ax.imshow(heat.numpy(), cmap="jet", alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def merge_subwords(tokenizer, input_ids, scores):
    """Gộp subword BPE của PhoBERT (token kết thúc bằng "@@" nối với token sau) thành từ,
    điểm của từ = tổng điểm các subword. Bỏ token đặc biệt."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special = set(tokenizer.all_special_tokens)
    words, word_scores, buf, buf_score = [], [], "", 0.0
    for tok, s in zip(tokens, scores):
        if tok in special:
            continue
        if tok.endswith("@@"):
            buf += tok[:-2]
            buf_score += float(s)
            continue
        words.append((buf + tok).replace("_", " "))
        word_scores.append(buf_score + float(s))
        buf, buf_score = "", 0.0
    if buf:
        words.append(buf.replace("_", " "))
        word_scores.append(buf_score)
    return words, word_scores


def words_to_html(words, scores):
    """Highlight từ theo điểm attention, hiển thị trong notebook bằng IPython.display.HTML."""
    s = np.asarray(scores, dtype=float)
    s = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s)
    spans = [
        f'<span style="background: rgba(255, 80, 0, {v:.2f}); padding: 1px 2px; border-radius: 3px">'
        f"{html.escape(w)}</span>"
        for w, v in zip(words, s)
    ]
    return '<div style="line-height: 2">' + " ".join(spans) + "</div>"
