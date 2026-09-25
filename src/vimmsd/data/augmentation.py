from torchvision import transforms


def build_image_augment(enabled: bool):
    """Augmentation nhẹ trên PIL image, chạy trước image processor của encoder.
    Không lật ngang vì nhiều ảnh là meme có chữ, lật sẽ làm chữ bị ngược."""
    if not enabled:
        return None
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])


class BackTranslator:
    """Back-translation vi -> en -> vi để sinh thêm caption cho lớp hiếm (text-sarcasm).
    Chạy offline một lần rồi lưu kết quả, không chạy trong vòng lặp train."""

    def __init__(
        self,
        vi2en="vinai/vinai-translate-vi2en-v2",
        en2vi="vinai/vinai-translate-en2vi-v2",
        device="cuda",
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device
        self.tok_vi2en = AutoTokenizer.from_pretrained(vi2en, src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained(vi2en).to(device).eval()
        self.tok_en2vi = AutoTokenizer.from_pretrained(en2vi, src_lang="en_XX")
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(en2vi).to(device).eval()

    def _translate(self, texts, tok, model, tgt_lang, **gen_kwargs):
        import torch

        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                decoder_start_token_id=tok.convert_tokens_to_ids(tgt_lang),
                max_length=256,
                **gen_kwargs,
            )
        return tok.batch_decode(out, skip_special_tokens=True)

    def __call__(self, texts, batch_size=16, num_return_sequences=1):
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            en = self._translate(chunk, self.tok_vi2en, self.model_vi2en, "en_XX", num_beams=5)
            vi = self._translate(
                en, self.tok_en2vi, self.model_en2vi, "vi_VN",
                do_sample=True, top_p=0.9, num_return_sequences=num_return_sequences,
            )
            results.extend(vi)
        return results
