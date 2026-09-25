"""Tiền xử lý ảnh thành text: chữ trong ảnh (OCR) và mô tả nội dung ảnh (VLM).

Chạy offline một lần cho toàn bộ ảnh (scripts/extract_image_text.py), lưu cache JSON
{"<tên thư mục ảnh>/<tên ảnh>": text}. Khi train, dataset đọc cache và ghép thành segment
thứ 2 của PhoBERT: <s> caption </s></s> chữ trong ảnh + mô tả ảnh </s>.
"""
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")

# Mô tả trung tính, không nhắc tới mỉa mai: tránh để VLM đưa phán đoán nhãn vào input của model,
# và không đọc lại chữ trong ảnh vì phần đó đã có OCR.
DESCRIBE_PROMPT = (
    "Mô tả ngắn gọn nội dung bức ảnh bằng tiếng Việt trong 1-2 câu: có ai hoặc cái gì, "
    "đang làm gì, bối cảnh và biểu cảm. Không chép lại chữ xuất hiện trong ảnh, không suy đoán ý nghĩa."
)


def image_key(image_dir, image_name):
    # không phụ thuộc đường dẫn tuyệt đối, để cache tạo trên Kaggle dùng được ở local/Colab
    return f"{Path(image_dir).name}/{image_name}"


def compose_image_text(ocr="", description=""):
    parts = []
    if ocr and ocr.strip():
        parts.append(f"Chữ trong ảnh: {ocr.strip()}.")
    if description and description.strip():
        parts.append(f"Mô tả ảnh: {description.strip()}")
    return " ".join(parts)


class OCRExtractor:
    """Wrapper PaddleOCR, hỗ trợ cả API 2.x (`.ocr`) và 3.x (`.predict`)."""

    def __init__(self, lang="vi", min_score=0.5, **kwargs):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(lang=lang, **kwargs)
        self.min_score = min_score

    def __call__(self, image_path) -> str:
        path = str(image_path)
        if hasattr(self.ocr, "predict"):
            pairs = []
            for r in self.ocr.predict(path):
                pairs.extend(zip(r["rec_texts"], r["rec_scores"]))
        else:
            results = self.ocr.ocr(path) or []
            pairs = [(line[1][0], line[1][1]) for page in results if page for line in page]
        return " ".join(text for text, score in pairs if score >= self.min_score)


VINTERN_MEAN, VINTERN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_vintern(model_name="5CD-AI/Vintern-1B-v3_5", dtype=torch.bfloat16):
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def vintern_pixel_values(image_path, dtype=torch.bfloat16, size=448):
    """1 tile 448x448 thay cho dynamic tiling của InternVL: đơn giản, ít VRAM, đủ cho mô tả tổng quát."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(VINTERN_MEAN, VINTERN_STD),
    ])
    return transform(Image.open(image_path).convert("RGB"))[None].to(dtype).cuda()


class VLMDescriber:
    """Sinh mô tả ảnh bằng VLM.

    backend="vintern": Vintern (5CD-AI), tối ưu cho tiếng Việt, 1B tham số chạy được trên T4.
    backend="hf": VLM chuẩn chat template của transformers (ví dụ Qwen/Qwen2.5-VL-3B-Instruct)."""

    def __init__(self, backend="vintern", model_name=None, prompt=DESCRIBE_PROMPT, max_new_tokens=96):
        self.backend = backend
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        if backend == "vintern":
            self.model, self.tokenizer = load_vintern(model_name or "5CD-AI/Vintern-1B-v3_5")
        elif backend == "hf":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_name = model_name or "Qwen/Qwen2.5-VL-3B-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
        else:
            raise ValueError(f"backend không hợp lệ: {backend}")

    @torch.no_grad()
    def __call__(self, image_path) -> str:
        if self.backend == "vintern":
            generation = dict(max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.3)
            out = self.model.chat(self.tokenizer, vintern_pixel_values(image_path), "<image>\n" + self.prompt,
                                  generation)
            return out.strip()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def list_images(image_dirs):
    return [p for d in image_dirs for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def build_image_text_cache(image_paths, out_path, extractor, save_every=100, desc="image text"):
    """Chạy `extractor(path) -> str` cho từng ảnh, lưu {image_key: text} ra JSON.
    Chạy tiếp được nếu bị ngắt (bỏ qua ảnh đã có trong file), lưu định kỳ mỗi `save_every` ảnh."""
    out_path = Path(out_path)
    cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [p for p in image_paths if image_key(p.parent, p.name) not in cache]
    logger.info("%s: %d ảnh đã có trong cache, còn %d ảnh", desc, len(image_paths) - len(todo), len(todo))

    for i, p in enumerate(tqdm(todo, desc=desc)):
        try:
            cache[image_key(p.parent, p.name)] = extractor(p)
        except Exception as e:  # noqa: BLE001 - một ảnh lỗi không nên dừng cả tiến trình
            logger.warning("%s lỗi ở %s: %s", desc, p, e)
            cache[image_key(p.parent, p.name)] = ""
        if (i + 1) % save_every == 0:
            _write_json(out_path, cache)
    _write_json(out_path, cache)
    return cache


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
