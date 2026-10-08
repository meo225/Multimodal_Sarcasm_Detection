"""Tiền xử lý ảnh thành text: chữ trong ảnh (OCR) và mô tả nội dung ảnh (VLM).

Chạy offline một lần cho toàn bộ ảnh (scripts/extract_image_text.py), lưu cache JSON
{"<tên thư mục ảnh>/<tên ảnh>": text}. Khi train, dataset đọc cache và ghép thành segment
thứ 2 của PhoBERT: <s> caption </s></s> chữ trong ảnh + mô tả ảnh </s>.
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from vimmsd.data.image_io import open_image_rgb

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


def reading_order(boxes):
    """Sắp các box (x0, y0, x1, y1) theo thứ tự đọc: từng dòng từ trên xuống, trong dòng từ trái sang.
    Hai box cùng dòng nếu tâm theo chiều dọc lệch nhau dưới nửa chiều cao box."""
    lines = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        center = (box[1] + box[3]) / 2
        if lines and abs(center - lines[-1][0]) < 0.5 * (box[3] - box[1]):
            lines[-1][1].append(box)
        else:
            lines.append((center, [box]))
    return [box for _, line in lines for box in sorted(line, key=lambda b: b[0])]


class OCRExtractor:
    """OCR hai bước: PaddleOCR (>= 3.0) phát hiện vùng chữ, VietOCR nhận dạng từng dòng.

    Không dùng phần nhận dạng của PaddleOCR vì bộ ký tự của các model rec (latin PP-OCRv3/v5, PP-OCRv6)
    thiếu chữ cái mang dấu thanh tiếng Việt (ạ, ế, ộ...): "thật làm phiền" bị đọc thành "tht làm phin".

    rec_model: "vgg_seq2seq" (mặc định, giải mã GRU, nhanh hơn nhiều) hoặc "vgg_transformer" (chậm hơn,
    chính xác hơn một chút theo README của VietOCR). Đổi model thì đổi tên file cache, không trộn hai model."""

    def __init__(self, rec_model="vgg_seq2seq", min_score=0.8, device=None, pad=2, **det_kwargs):
        from paddleocr import TextDetection
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        self.detector = TextDetection(**det_kwargs)
        rec_cfg = Cfg.load_config_from_name(rec_model)
        rec_cfg["device"] = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        rec_cfg["cnn"]["pretrained"] = False  # trọng số VietOCR đã gồm backbone, không cần tải thêm VGG ImageNet
        self.recognizer = Predictor(rec_cfg)
        # độ tin cậy của VietOCR: dòng chữ rõ thường >= 0.85, vùng không phải chữ hoặc chữ không phải Latin < 0.8
        self.min_score = min_score
        self.pad = pad

    def _line_crops(self, image_path):
        # cùng cách đọc với đường A: frame đầu, sửa hướng EXIF, nền trắng cho vùng trong suốt
        image = open_image_rgb(image_path)
        # đưa mảng BGR thay vì đường dẫn: detector và bước crop dùng chung một ảnh (cùng frame, cùng chiều xoay)
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        polys = [np.asarray(p) for r in self.detector.predict(bgr) for p in r["dt_polys"]]
        boxes = reading_order([(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polys])
        return [image.crop((max(0, x0 - self.pad), max(0, y0 - self.pad), x1 + self.pad, y1 + self.pad))
                for x0, y0, x1, y1 in boxes]

    def extract_many(self, image_paths):
        """OCR nhiều ảnh một lượt: phát hiện vùng chữ từng ảnh, rồi nhận dạng dòng chữ của CẢ nhóm ảnh trong một
        lần gọi VietOCR. Mỗi ảnh chỉ có vài dòng, gộp nhiều ảnh thì GPU chạy batch lớn và nhanh hơn hẳn."""
        crops, owners = [], []
        for k, path in enumerate(image_paths):
            for crop in self._line_crops(path):
                crops.append(crop)
                owners.append(k)
        lines = [[] for _ in image_paths]
        if crops:
            texts, scores = self.recognizer.predict_batch(crops, return_prob=True)
            for k, text, score in zip(owners, texts, scores):
                # score NaN (VietOCR không tự tin ký tự nào) so sánh >= luôn False nên dòng đó bị bỏ
                if score >= self.min_score and text.strip():
                    lines[k].append(text.strip())
        # mỗi dòng một hàng, để bước làm sạch OCR (preprocessing.clean_ocr) lọc được từng dòng rác
        return ["\n".join(ls) for ls in lines]

    def __call__(self, image_path) -> str:
        return self.extract_many([image_path])[0]


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
    return transform(open_image_rgb(image_path))[None].to(dtype).cuda()


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
        image = open_image_rgb(image_path)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def list_images(image_dirs):
    return [p for d in image_dirs for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def build_image_text_cache(image_paths, out_path, extractor, save_every=100, desc="image text",
                           max_consecutive_failures=20, batch_size=1):
    """Chạy `extractor(path) -> str` cho từng ảnh, lưu {image_key: text} ra JSON.
    Chạy tiếp được nếu bị ngắt (bỏ qua ảnh đã có trong file), lưu định kỳ khoảng mỗi `save_every` ảnh.
    `batch_size` > 1 và extractor có `extract_many(paths) -> list[str]` (OCRExtractor): xử lý theo nhóm ảnh;
    nhóm nào lỗi thì chạy lại từng ảnh của nhóm đó để chỉ bỏ qua đúng ảnh lỗi.

    Ảnh lỗi KHÔNG được ghi vào cache (chuỗi rỗng chỉ có nghĩa "ảnh không có text"), nên lần chạy sau
    sẽ thử lại. Lỗi `max_consecutive_failures` ảnh liên tiếp thì dừng hẳn: đó là lỗi môi trường
    (hết VRAM, thiếu thư viện, model hỏng), chạy tiếp chỉ tạo ra cache thiếu."""
    out_path = Path(out_path)
    cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [p for p in image_paths if image_key(p.parent, p.name) not in cache]
    logger.info("%s: %d ảnh đã có trong cache, còn %d ảnh", desc, len(image_paths) - len(todo), len(todo))
    if batch_size <= 1 or not hasattr(extractor, "extract_many"):
        batch_size = 1

    failed, streak, since_save = [], 0, 0
    bar = tqdm(total=len(todo), desc=desc)
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        results = None
        if batch_size > 1:
            try:
                results = extractor.extract_many(chunk)
            except Exception as e:  # noqa: BLE001 - chạy lại từng ảnh bên dưới để tìm đúng ảnh lỗi
                logger.warning("%s lỗi ở nhóm %d ảnh (%r), chạy lại từng ảnh", desc, len(chunk), e)
        for k, p in enumerate(chunk):
            try:
                cache[image_key(p.parent, p.name)] = results[k] if results is not None else extractor(p)
                streak = 0
            except Exception as e:  # noqa: BLE001 - một ảnh lỗi không nên dừng cả tiến trình
                logger.warning("%s lỗi ở %s: %r", desc, p, e)
                failed.append(p)
                streak += 1
                if streak >= max_consecutive_failures:
                    _write_json(out_path, cache)
                    raise RuntimeError(f"{desc}: {streak} ảnh lỗi liên tiếp, dừng lại. Lỗi cuối: {e!r}") from e
        bar.update(len(chunk))
        since_save += len(chunk)
        if since_save >= save_every:
            _write_json(out_path, cache)
            since_save = 0
    bar.close()
    _write_json(out_path, cache)
    if failed:
        logger.warning("%s: %d/%d ảnh lỗi, chưa có trong cache (chạy lại để thử lại)", desc, len(failed), len(todo))
    return cache


def shard_cache_name(name, shard, num_shards):
    """"ocr_v2.json" -> "ocr_v2.shard0of2.json": file cache riêng của từng tiến trình khi chạy song song."""
    path = Path(name)
    return f"{path.stem}.shard{shard}of{num_shards}{path.suffix}"


def merge_image_text_caches(out_path, shard_paths):
    """Gộp các file cache shard (và file cache cũ ở `out_path` nếu có) thành một file. Trả về dict đã gộp."""
    out_path = Path(out_path)
    merged = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for p in shard_paths:
        merged.update(json.loads(Path(p).read_text(encoding="utf-8")))
    _write_json(out_path, merged)
    return merged


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
