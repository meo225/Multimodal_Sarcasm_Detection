import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# Chỉ gồm các từ viết tắt có nghĩa rõ ràng; bỏ qua từ đa nghĩa như "m", "t", "v", "bn".
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "hok": "không", "hem": "không",
    "hông": "không", "khum": "không", "kg": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "j": "gì", "ji": "gì",
    "vs": "với", "zới": "với",
    "mn": "mọi người", "mng": "mọi người",
    "ng": "người",
    "cx": "cũng", "cug": "cũng", "cũg": "cũng",
    "r": "rồi", "rùi": "rồi", "ròi": "rồi",
    "ntn": "như thế nào",
    "trc": "trước",
    "wa": "quá", "qá": "quá", "qa": "quá",
    "mik": "mình", "mk": "mình", "mh": "mình",
    "lm": "làm",
    "bít": "biết", "bik": "biết", "bjt": "biết",
    "đag": "đang", "dag": "đang",
    "z": "vậy", "zậy": "vậy", "vại": "vậy",
    "thik": "thích", "thjk": "thích",
    "iu": "yêu",
    "nhìu": "nhiều",
    "ak": "à",
    "ib": "nhắn tin",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
    "hnay": "hôm nay", "hqua": "hôm qua",
    "ae": "anh em",
    "nma": "nhưng mà", "nhma": "nhưng mà",
    "bh": "bây giờ", "bjo": "bây giờ",
    "sn": "sinh nhật",
    "ny": "người yêu",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    # dữ liệu mạng xã hội trộn lẫn dạng dựng sẵn và tổ hợp của dấu tiếng Việt
    return unicodedata.normalize("NFC", text)


def _base_letter(ch: str) -> str:
    return unicodedata.normalize("NFD", ch)[0].lower()


def collapse_repeats(text: str) -> str:
    """Rút gọn chuỗi >= 3 chữ cái cùng gốc (bỏ qua dấu) về chữ đầu: "quáaaa" -> "quá", "đẹppppp" -> "đẹp".
    Không đụng tới số ("1000") hay chữ lặp 2 lần hợp lệ ("xoong")."""
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
    raise ValueError(f"emoji mode không hợp lệ: {mode}")


@lru_cache(maxsize=1)
def _word_tokenize():
    from underthesea import word_tokenize

    return word_tokenize


def word_segment(text: str) -> str:
    # PhoBERT được pretrain trên văn bản đã tách từ, từ ghép nối bằng "_" (ví dụ "mạng_xã_hội")
    return _word_tokenize()(text, format="text")


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_teencode_: bool = True,
    emoji_mode: str = "demojize",
    word_segment_: bool = True,
) -> str:
    text = normalize_unicode(text or "")
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
    """Tiền xử lý toàn bộ caption một lần và cache ra đĩa (tách từ underthesea khá chậm)."""

    def __init__(self, lowercase=False, normalize_teencode=True, emoji="demojize", word_segment=True):
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
