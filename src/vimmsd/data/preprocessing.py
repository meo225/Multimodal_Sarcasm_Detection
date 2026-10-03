import hashlib
import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# Bảng ánh xạ teencode và từ viết tắt tiếng Việt
TEENCODE = {
    # Phủ định
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "hok": "không",
    "hem": "không", "hông": "không", "hổng": "không", "hơm": "không", "hăm": "không",
    "khum": "không", "kô": "không", "chx": "chưa",

    # Động từ và trạng thái
    "dc": "được", "đc": "được",
    "lm": "làm", "bít": "biết", "bik": "biết", "bjt": "biết",
    "đag": "đang", "dag": "đang", "thik": "thích", "thjk": "thích",
    "iu": "yêu", "ib": "nhắn tin", "nt": "nhắn tin", "tl": "trả lời",
    "hỉu": "hiểu",

    # Danh từ và đại từ
    "j": "gì", "ji": "gì",
    "vs": "với", "zới": "với",
    "mn": "mọi người", "mng": "mọi người",
    "ng": "người", "mik": "mình", "mh": "mình",
    "ae": "anh em", "ny": "người yêu", "cr": "crush", "gđ": "gia đình",
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
    "thui": "thôi", "lun": "luôn", "nx": "nữa",
    "đou": "đâu", "đâuu": "đâu", "hẻ": "hả",
    "đr": "đúng rồi", "đug": "đúng",
    "uk": "ừ", "uhm": "ừ", "ukm": "ừ",
    "oki": "ok", "okie": "ok", "okela": "ok",
    "klq": "không liên quan", "tg": "thời gian",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\xa0\u202a-\u202e]")


def normalize_unicode(text: str) -> str:
    # Chuẩn hóa về dạng Unicode NFC
    return unicodedata.normalize("NFC", text)


def _base_letter(ch: str) -> str:
    return unicodedata.normalize("NFD", ch)[0].lower()


def collapse_repeats(text: str) -> str:
    """Rút gọn chuỗi có từ 3 ký tự lặp liên tiếp cùng gốc chữ cái về một ký tự."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        j = i + 1
        if ch.isalpha():
            base = _base_letter(ch)
            while j < len(text) and text[j].isalpha() and _base_letter(text[j]) == base:
                j += 1
        out.append(ch if j - i >= 3 else text[i:j])
        i = j
    return "".join(out)


def normalize_teencode(text: str) -> str:
    def repl(m):
        w = m.group(0)
        return TEENCODE.get(w.lower(), w)

    return WORD_RE.sub(repl, text)


def handle_emoji(text: str, mode: str) -> str:
    if mode == "keep":
        return text
    import emoji

    if mode == "remove":
        return emoji.replace_emoji(text, replace=" ")
    if mode == "demojize":
        text = emoji.demojize(text, delimiters=(" ", " "))
        return text.replace("_", " ")
    raise ValueError(f"Chế độ emoji không hợp lệ: {mode}")


def extract_emojis(text: str) -> list[str]:
    """Return emojis in source order, preserving repeated occurrences."""
    import emoji
    return [item["emoji"] for item in emoji.emoji_list(text or "")]


@lru_cache(maxsize=1)
def _word_tokenize():
    from underthesea import word_tokenize

    return word_tokenize


def word_segment(text: str) -> str:
    # Tách từ tiếng Việt theo định dạng từ ghép nối bằng gạch dưới cho PhoBERT
    return _word_tokenize()(text, format="text")


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_teencode_: bool = True,
    emoji_mode: str = "demojize",
    word_segment_: bool = True,
) -> str:
    text = html.unescape(text or "")
    text = INVISIBLE_CHARS_RE.sub(" ", text)
    text = normalize_unicode(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = collapse_repeats(text)
    text = handle_emoji(text, emoji_mode)
    if lowercase:
        text = text.lower()
    if normalize_teencode_:
        text = normalize_teencode(text)
    text = SPACE_RE.sub(" ", text).strip()
    if word_segment_ and text:
        text = word_segment(text)
    return text


class TextPreprocessor:
    """Xử lý hàng loạt và lưu trữ kết quả tiền xử lý văn bản ra đĩa đệm."""

    def __init__(self, lowercase=False, normalize_teencode=True, emoji="demojize", word_segment=True, include_emoji_explanation=False):
        self.include_emoji_explanation = include_emoji_explanation
        self.kwargs = dict(
            lowercase=lowercase,
            normalize_teencode_=normalize_teencode,
            emoji_mode=emoji,
            word_segment_=word_segment,
        )

    @classmethod
    def from_config(cls, text_cfg):
        return cls(**dict(text_cfg))

    def __call__(self, text: str) -> str:
        return clean_text(text, **self.kwargs)

    def _cache_file(self, cache_dir):
        key = hashlib.md5(json.dumps(self.kwargs, sort_keys=True).encode()).hexdigest()[:8]
        return Path(cache_dir) / f"text_cache_{key}.json"

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
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
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
