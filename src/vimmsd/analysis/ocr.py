import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Wrapper PaddleOCR, hỗ trợ cả API 2.x (`.ocr`) và 3.x (`.predict`)."""

    def __init__(self, lang="vi", min_score=0.5, **kwargs):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(lang=lang, **kwargs)
        self.min_score = min_score

    def __call__(self, image_path) -> str:
        path = str(image_path)
        if hasattr(self.ocr, "predict"):
            results = self.ocr.predict(path)
            pairs = []
            for r in results:
                pairs.extend(zip(r["rec_texts"], r["rec_scores"]))
        else:
            results = self.ocr.ocr(path) or []
            pairs = [(line[1][0], line[1][1]) for page in results if page for line in page]
        return " ".join(text for text, score in pairs if score >= self.min_score)


def build_ocr_cache(image_dirs, out_path, extractor=None, exts=(".jpg", ".jpeg", ".png", ".webp", ".gif")):
    """Chạy OCR cho mọi ảnh trong các thư mục, lưu {"<tên thư mục>/<tên ảnh>": text} ra JSON.
    Có thể chạy tiếp nếu bị ngắt giữa chừng (bỏ qua ảnh đã có trong file)."""
    extractor = extractor or OCRExtractor()
    out_path = Path(out_path)
    cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    paths = [p for d in image_dirs for p in sorted(Path(d).iterdir()) if p.suffix.lower() in exts]
    todo = [p for p in paths if f"{p.parent.name}/{p.name}" not in cache]
    for i, p in enumerate(tqdm(todo, desc="OCR")):
        try:
            cache[f"{p.parent.name}/{p.name}"] = extractor(p)
        except Exception as e:  # noqa: BLE001 - một ảnh lỗi không nên dừng cả tiến trình OCR
            logger.warning("OCR lỗi ở %s: %s", p, e)
            cache[f"{p.parent.name}/{p.name}"] = ""
        if (i + 1) % 200 == 0:
            _write(out_path, cache)
    _write(out_path, cache)
    return cache


def _write(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
