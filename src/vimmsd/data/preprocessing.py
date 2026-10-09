import html
import json
import os
import re
import unicodedata
import urllib.request
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from vimmsd.data.emoji_vi import EMOJI_VI

# Tăng mỗi khi sửa code làm sạch: là một phần của key cache, để không dùng lại kết quả của pipeline cũ.
PREPROCESS_VERSION = 2

# Bảng ánh xạ teencode và từ viết tắt tiếng Việt.
# Đã bỏ các mục đa nghĩa hay thay nhầm từ chuẩn: "hổng" (lỗ hổng), "hăm" (hăm dọa), "thui" (tối thui),
# "bít" (bít tắc), "uk" (nước Anh), "tg" (tác giả), "cr" (credit), "nt" (nhiều nghĩa).
TEENCODE = {
    # Phủ định
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "hok": "không",
    "hem": "không", "hông": "không", "hơm": "không",
    "khum": "không", "kô": "không", "chx": "chưa",

    # Động từ và trạng thái
    "dc": "được", "đc": "được",
    "lm": "làm", "bik": "biết", "bjt": "biết",
    "đag": "đang", "dag": "đang", "thik": "thích", "thjk": "thích",
    "iu": "yêu", "ib": "nhắn tin", "tl": "trả lời",
    "hỉu": "hiểu",

    # Danh từ và đại từ
    "j": "gì", "ji": "gì",
    "vs": "với", "zới": "với",
    "mn": "mọi người", "mng": "mọi người",
    "ng": "người", "mik": "mình", "mh": "mình",
    "ae": "anh em", "ny": "người yêu", "gđ": "gia đình",
    "cmt": "bình luận", "stt": "trạng thái",
    "acc": "tài khoản", "sđt": "số điện thoại",

    # Liên từ và từ cảm thán
    "cx": "cũng", "cug": "cũng", "cũg": "cũng",
    "rùi": "rồi", "ròi": "rồi",
    "ntn": "như thế nào", "trc": "trước",
    "wa": "quá", "qá": "quá", "qa": "quá",
    "z": "vậy", "zậy": "vậy", "vại": "vậy",
    "nhìu": "nhiều", "ak": "à",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
    "hnay": "hôm nay", "hqua": "hôm qua",
    "nma": "nhưng mà", "nhma": "nhưng mà",
    "bjo": "bây giờ",
    "lun": "luôn", "nx": "nữa",
    "đou": "đâu", "đâuu": "đâu", "hẻ": "hả",
    "đr": "đúng rồi", "đug": "đúng",
    "uhm": "ừ", "ukm": "ừ",
    "oki": "ok", "okie": "ok", "okela": "ok",
    "klq": "không liên quan",
}

# Khóa ngắn trùng với chữ viết tắt hoặc tên riêng khi viết hoa toàn bộ (Gen Z, K-Pop, súng AK, KH):
# chỉ thay khi không viết hoa toàn bộ. Các khóa khác thay cả khi viết hoa (KO, ĐC trong meme).
CASE_SENSITIVE_TEENCODE = {"k", "z", "j", "ak", "kh", "lm", "ng"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
# zero-width, BOM, ký tự định hướng: xóa hẳn, vì thay bằng khoảng trắng sẽ tách đôi từ ("kh​ông")
INVISIBLE_CHARS_RE = re.compile(r"[​‌‍‎‏﻿‪-‮]")
EMOJI_MODIFIERS_RE = re.compile("[️\U0001F3FB-\U0001F3FF]")  # variation selector, màu da

SEGMENTERS = ("vncorenlp", "underthesea", "none")


def normalize_unicode(text: str) -> str:
    # Chuẩn hóa về dạng Unicode NFC
    return unicodedata.normalize("NFC", text)


def remove_invisible_chars(text: str) -> str:
    return INVISIBLE_CHARS_RE.sub("", text).replace("\xa0", " ")


def _base_letter(ch: str) -> str:
    return unicodedata.normalize("NFD", ch)[0].lower()


def _collapse_word(word: str) -> str:
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        j = i + 1
        if ch.isalpha() and ch.islower():
            base = _base_letter(ch)
            while j < len(word) and word[j].isalpha() and word[j].islower() and _base_letter(word[j]) == base:
                j += 1
        out.append(ch if j - i >= 3 else word[i:j])
        i = j
    return "".join(out)


def collapse_repeats(text: str) -> str:
    """Rút gọn chuỗi từ 3 chữ cái thường lặp liên tiếp cùng gốc (bỏ qua dấu) về một chữ: "quáaaa" -> "quá".
    Bỏ qua token viết hoa toàn bộ để giữ từ viết tắt (PCCC, CCCD, VIII); đánh đổi là "GOALLLL" cũng giữ nguyên."""
    return WORD_RE.sub(lambda m: m.group(0) if m.group(0).isupper() else _collapse_word(m.group(0)), text)


def normalize_teencode(text: str) -> str:
    def repl(m):
        w = m.group(0)
        key = w.lower()
        if key in CASE_SENSITIVE_TEENCODE and w.isupper():
            return w
        return TEENCODE.get(key, w)

    return WORD_RE.sub(repl, text)


def _emoji_to_vietnamese(chars, data):
    meaning = EMOJI_VI.get(EMOJI_MODIFIERS_RE.sub("", chars))
    if meaning is None:
        meaning = data["en"].strip(":").replace("_", " ")
    return f" {meaning} "


def handle_emoji(text: str, mode: str) -> str:
    """keep: giữ nguyên; remove: xóa; demojize: emoji trong EMOJI_VI thành nghĩa tiếng Việt,
    emoji khác thành tên tiếng Anh. Xử lý cả chuỗi emoji ghép bằng ZWJ như một emoji."""
    if mode == "keep":
        return text
    import emoji

    if mode == "remove":
        return emoji.replace_emoji(text, replace=" ")
    if mode == "demojize":
        return emoji.replace_emoji(text, replace=_emoji_to_vietnamese)
    raise ValueError(f"Chế độ emoji không hợp lệ: {mode}")


def extract_emojis(text: str) -> list:
    """Danh sách emoji theo thứ tự xuất hiện, giữ cả các lần lặp. Emoji ghép bằng ZWJ tính là một emoji.
    Dùng trên caption THÔ cho các đặc trưng đếm emoji (mục E0 trong ghi chú pipeline text)."""
    import emoji

    return [item["emoji"] for item in emoji.emoji_list(text or "")]


def clean_ocr(ocr: str, caption: str = "", min_chars=3, max_similarity=0.9) -> str:
    """Làm sạch chữ OCR trước khi ghép vào segment 2: bỏ dòng có dưới `min_chars` chữ/số hoặc không có chữ cái
    nào (số lẻ, ký tự rác), và bỏ cả OCR nếu gần trùng caption (tỉ lệ giống nhau > `max_similarity`)."""
    lines = []
    for line in (ocr or "").splitlines():
        line = line.strip()
        alnum = [ch for ch in line if ch.isalnum()]
        if len(alnum) >= min_chars and any(ch.isalpha() for ch in alnum):
            lines.append(line)
    text = " ".join(lines)
    if text and caption and SequenceMatcher(None, text.lower(), caption.lower()).ratio() > max_similarity:
        return ""
    return text


VNCORENLP_DIR = Path(os.environ.get("VNCORENLP_DIR", Path.home() / ".cache" / "vncorenlp"))
VNCORENLP_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
VNCORENLP_FILES = ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr")



@lru_cache(maxsize=1)
def _vncorenlp():
    """RDRSegmenter của VnCoreNLP, công cụ tách từ PhoBERT dùng khi pretrain. Cần Java (JDK/JRE >= 8).
    Chỉ tải jar và model tách từ (không tải POS/NER/parse như `py_vncorenlp.download_model`)."""
    import py_vncorenlp

    save_dir = VNCORENLP_DIR.resolve()
    for name in VNCORENLP_FILES:
        path = save_dir / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(VNCORENLP_URL + name, path)
    cwd = os.getcwd()
    try:
        return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))
    finally:
        os.chdir(cwd)  # py_vncorenlp chdir vào save_dir khi khởi tạo và không trả lại


@lru_cache(maxsize=1)
def _underthesea():
    from underthesea import word_tokenize

    return word_tokenize


def _resolve_segmenter(segmenter) -> str:
    # tương thích ngược với config/notebook cũ dùng True/False
    if segmenter is True:
        return "vncorenlp"
    if segmenter in (False, None):
        return "none"
    if segmenter not in SEGMENTERS:
        raise ValueError(f"word_segment không hợp lệ: {segmenter!r}, chọn một trong {SEGMENTERS}")
    return segmenter


def word_segment(text: str, segmenter="vncorenlp") -> str:
    """Tách từ, nối từ ghép bằng gạch dưới ("mạng_xã_hội"). PhoBERT dùng VnCoreNLP;
    ViSoBERT pretrain trên văn bản thô nên dùng "none"."""
    segmenter = _resolve_segmenter(segmenter)
    if segmenter == "vncorenlp":
        return " ".join(_vncorenlp().word_segment(text))
    if segmenter == "underthesea":
        return _underthesea()(text, format="text")
    return text


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_teencode_: bool = True,
    emoji_mode: str = "demojize",
    word_segment_="vncorenlp",
) -> str:
    text = html.unescape(text or "")
    text = normalize_unicode(text)
    # emoji trước bước xóa ký tự vô hình: ZWJ (U+200D) nối các emoji ghép như 🤦‍♂️
    text = handle_emoji(text, emoji_mode)
    text = remove_invisible_chars(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = collapse_repeats(text)
    if normalize_teencode_:
        text = normalize_teencode(text)
    if lowercase:
        text = text.lower()
    text = SPACE_RE.sub(" ", text).strip()
    if text:
        text = word_segment(text, word_segment_)
    return text


class TextPreprocessor:
    """Xử lý hàng loạt và lưu trữ kết quả tiền xử lý văn bản ra đĩa đệm."""

    def __init__(self, lowercase=False, normalize_teencode=True, emoji="demojize", word_segment="vncorenlp", include_emoji_explanation=False):
        self.include_emoji_explanation = include_emoji_explanation
        self.kwargs = dict(
            lowercase=lowercase,
            normalize_teencode_=normalize_teencode,
            emoji_mode=emoji,
            word_segment_=_resolve_segmenter(word_segment),
        )

    @classmethod
    def from_config(cls, text_cfg):
        raw = dict(text_cfg)
        # cache_dir là đường dẫn file cache, không phải tham số chuẩn hóa câu
        allowed = {"lowercase", "normalize_teencode", "emoji", "word_segment", "include_emoji_explanation"}
        return cls(**{k: raw[k] for k in allowed if k in raw})

    def __call__(self, text: str) -> str:
        return clean_text(text, **self.kwargs)

    def _cache_file(self, cache_dir):
        return Path(cache_dir) / "01a_text_preprocessing.json"

    def process_all(self, texts, cache_dir=None):
        cache = {}
        cache_file = self._cache_file(cache_dir) if cache_dir else None
        if cache_file and cache_file.exists():
            cache = json.loads(cache_file.read_text(encoding="utf-8"))

        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        for t in missing:
            cache[t] = self(t)

        if cache_file and missing:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return [cache[t] for t in texts]

    def process_record(self, record: dict, emoji_explanation_text: str = "") -> dict:
        processed = dict(record)
        processed["caption_processed"] = self(record.get("caption", ""))
        if self.include_emoji_explanation:
            processed["emoji_explanation"] = emoji_explanation_text.strip()
        return processed

    def process_records(self, records, emoji_annotations=None):
        annotations = emoji_annotations or {}
        if isinstance(records, dict):
            return {key: self.process_record(value, annotations.get(str(key), "")) for key, value in records.items()}
        return [self.process_record(value, annotations.get(str(index), "")) for index, value in enumerate(records)]
