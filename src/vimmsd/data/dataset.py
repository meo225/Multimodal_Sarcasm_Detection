import json
import logging
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from vimmsd.data.augmentation import build_image_augment
from vimmsd.data.image_io import PadToSquare, encoder_mean_color, open_image_rgb
from vimmsd.data.image_text import compose_image_text, image_key
from vimmsd.data.preprocessing import TextPreprocessor, clean_ocr

logger = logging.getLogger(__name__)

IGNORE_LABEL = -1  # mẫu không có nhãn (public test)


def load_records(json_path, image_dir, label2id, image_texts=None):
    """Đọc file annotation ViMMSD: {id: {"image", "caption", "label"}} (hoặc list các dict).
    Trả về list dict thống nhất: id, image_path, caption, label (int, -1 nếu không có nhãn).
    image_texts: {"ocr": cache, "description": cache} tạo bởi scripts/extract_image_text.py."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.items() if isinstance(raw, dict) else ((str(i), r) for i, r in enumerate(raw))

    image_texts = image_texts or {}
    records, missing = [], Counter()
    for sid, r in items:
        label = r.get("label")
        rec = {
            "id": str(sid),
            "image_path": str(Path(image_dir) / r["image"]),
            "caption": r.get("caption", ""),
            "label": label2id[label] if label is not None else IGNORE_LABEL,
        }
        key = image_key(image_dir, r["image"])
        for field, cache in image_texts.items():
            missing[field] += key not in cache
            rec[field] = cache.get(key, "")
        records.append(rec)

    # ảnh không có trong cache khác với ảnh có text rỗng: thiếu nghĩa là chưa chạy (hoặc chạy lỗi) bước trích xuất
    for field, n in missing.items():
        if n and n == len(records):
            raise ValueError(f"cache '{field}' không có ảnh nào của {json_path} (thư mục ảnh {Path(image_dir).name}): "
                             "chạy scripts/extract_image_text.py cho split này hoặc kiểm tra data.image_text.dir")
        if n:
            logger.warning("cache '%s' thiếu %d/%d ảnh của %s, các mẫu này không có text từ ảnh",
                           field, n, len(records), json_path)
    return records


def split_records(records, val_ratio, test_ratio, seed):
    """Chia stratified train/val/test từ tập train có nhãn (public test của ViMMSD không có nhãn)."""
    labels = [r["label"] for r in records]
    holdout = val_ratio + test_ratio
    train, rest, _, rest_labels = train_test_split(
        records, labels, test_size=holdout, stratify=labels, random_state=seed
    )
    if test_ratio == 0:
        return train, rest, []
    val, test = train_test_split(
        rest, test_size=test_ratio / holdout, stratify=rest_labels, random_state=seed
    )
    return train, val, test


class ViMMSDDataset(Dataset):
    def __init__(self, records, text_preprocessor=None, image_transform=None,
                 load_image=True, cache_dir=None, use_image_text=False, missing_image_color=(0, 0, 0)):
        self.records = records
        self.image_transform = image_transform
        self.load_image = load_image
        self.missing_image_color = tuple(missing_image_color)

        captions = [r["caption"] for r in records]
        image_texts = [compose_image_text(clean_ocr(r.get("ocr", ""), r["caption"]), r.get("description", ""))
                       for r in records]
        if text_preprocessor is not None:
            captions = text_preprocessor.process_all(captions, cache_dir=cache_dir)
            if use_image_text:
                image_texts = text_preprocessor.process_all(image_texts, cache_dir=cache_dir)
        self.texts = captions
        self.image_texts = image_texts if use_image_text else None
        self._bad_images = 0

    def __len__(self):
        return len(self.records)

    @property
    def labels(self):
        return [r["label"] for r in self.records]

    def _open_image(self, path):
        """Trả về (ảnh, có lỗi hay không). Ảnh lỗi được thay bằng ảnh một màu `missing_image_color`
        (mean của encoder, sau normalize gần 0) và đánh cờ `img_missing` để đếm được số mẫu bị ảnh hưởng."""
        try:
            return open_image_rgb(path), False
        except (OSError, ValueError) as e:
            self._bad_images += 1
            if self._bad_images <= 5:
                logger.warning("không đọc được ảnh %s (%s), thay bằng ảnh một màu và đặt img_missing=1", path, e)
            return Image.new("RGB", (224, 224), self.missing_image_color), True

    def __getitem__(self, idx):
        rec = self.records[idx]
        item = {"id": rec["id"], "text": self.texts[idx], "label": rec["label"]}
        if self.image_texts is not None:
            item["image_text"] = self.image_texts[idx]
        if self.load_image:
            img, missing = self._open_image(rec["image_path"])
            if self.image_transform is not None:
                img = self.image_transform(img)
            item["image"] = img
            item["img_missing"] = int(missing)
        return item


class ViMMSDCollator:
    """Tokenize text + xử lý ảnh theo batch (padding động theo câu dài nhất trong batch).

    Khi có text từ ảnh (2 segment), caption được cắt trước còn tối đa `max_caption_length` token, phần còn
    thiếu chỗ chỉ cắt ở segment 2 (`only_second`). Không dùng `only_second` một mình: nếu riêng caption đã
    dài hơn `max_length`, tokenizer không cắt gì và trả về chuỗi dài hơn giới hạn của PhoBERT."""

    def __init__(self, tokenizer=None, image_processor=None, max_length=128, max_caption_length=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.max_caption_length = max_caption_length or max_length

    def _clip_caption(self, text):
        # only_second chỉ cắt được khi segment 2 dài hơn số token cần bỏ, tức caption + token đặc biệt < max_length
        limit = min(self.max_caption_length, self.max_length - self.tokenizer.num_special_tokens_to_add(pair=True) - 1)
        tokens = self.tokenizer.tokenize(text)
        return text if len(tokens) <= limit else self.tokenizer.convert_tokens_to_string(tokens[:limit])

    def __call__(self, items):
        batch = {
            "ids": [it["id"] for it in items],
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
        }
        if self.tokenizer is not None:
            texts = [it["text"] for it in items]
            if "image_text" in items[0]:
                # 2 segment: <s> caption </s></s> chữ trong ảnh + mô tả ảnh </s>
                enc = self.tokenizer([self._clip_caption(t) for t in texts], [it["image_text"] for it in items],
                                     padding=True, truncation="only_second", max_length=self.max_length,
                                     return_tensors="pt")
            else:
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            batch["input_ids"] = enc["input_ids"]
            batch["attention_mask"] = enc["attention_mask"]
        if self.image_processor is not None:
            images = [it["image"] for it in items]
            batch["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        if "img_missing" in items[0]:
            batch["img_missing"] = torch.tensor([it["img_missing"] for it in items], dtype=torch.long)
        return batch


def text_cache_dir(cfg):
    """Thư mục chứa 01a_text_preprocessing.json. None trong config thì dùng paths.cache_dir."""
    text_cfg = cfg.data.get("text") or {}
    return text_cfg.get("cache_dir") or cfg.paths.get("cache_dir")


def collect_captions(cfg):
    """Mọi caption trong train / public test / private test. File không có thì bỏ qua."""
    data_dir = Path(cfg.paths.data_dir)
    names = [cfg.data.train_json]
    for key in ("public_test_json", "private_test_json"):
        name = cfg.data.get(key)
        if name:
            names.append(name)
    captions, used = [], []
    for name in names:
        path = data_dir / name
        if not path.exists():
            logger.warning("bỏ qua %s: không thấy file", path)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.values() if isinstance(raw, dict) else raw
        captions.extend(str(item.get("caption", "")) for item in items)
        used.append(name)
    return captions, used


def build_text_cache(cfg):
    """Chuẩn hóa toàn bộ caption và ghi 01a_text_preprocessing.json vào paths.cache_dir.

    Notebook train đọc lại file này qua TextPreprocessor.process_all, nên không xử lý lại từ đầu.
    """
    pre = TextPreprocessor.from_config(cfg.data.text)
    out_dir = Path(cfg.paths.cache_dir)
    captions, used = collect_captions(cfg)
    pre.process_all(captions, cache_dir=out_dir)
    return pre._cache_file(out_dir), len(captions), used


def load_image_texts(cfg):
    """Đọc cache OCR / mô tả VLM theo `data.image_text` trong config. Trả về {} nếu không dùng."""
    it_cfg = cfg.data.get("image_text") or {}
    cache_dir = Path(it_cfg.get("dir") or cfg.paths.cache_dir)
    texts = {}
    for field, flag, name in (("ocr", "use_ocr", "ocr_cache"), ("description", "use_description", "description_cache")):
        if it_cfg.get(flag):
            path = cache_dir / it_cfg[name]
            if not path.exists():
                raise FileNotFoundError(f"thiếu {path}: chạy scripts/extract_image_text.py (notebooks/01b) trước")
            texts[field] = json.loads(path.read_text(encoding="utf-8"))
    return texts


def uses_image_text(cfg):
    it_cfg = cfg.data.get("image_text") or {}
    return bool(it_cfg.get("use_ocr") or it_cfg.get("use_description"))


def load_all_records(cfg):
    """Trả về dict split -> records: train/val/test (chia từ tập có nhãn) và public_test (không nhãn)."""
    data_dir = Path(cfg.paths.data_dir)
    label2id = {l: i for i, l in enumerate(cfg.data.labels)}
    image_texts = load_image_texts(cfg)

    labeled = load_records(data_dir / cfg.data.train_json, data_dir / cfg.data.train_image_dir,
                           label2id, image_texts)
    train, val, test = split_records(labeled, cfg.data.val_ratio, cfg.data.test_ratio, cfg.seed)
    splits = {"train": train, "val": val, "test": test}

    public_json = cfg.data.get("public_test_json")
    if public_json and (data_dir / public_json).exists():
        splits["public_test"] = load_records(data_dir / public_json,
                                             data_dir / cfg.data.public_test_image_dir,
                                             label2id, image_texts)
    return splits


def build_image_transform(cfg, train, fill):
    """Đường A trước image processor: pad vuông (A2, mặc định) rồi augment nhẹ (A3, chỉ khi train).
    `data.image_resize: crop` bỏ bước pad, để processor center-crop như code cũ (dùng cho ablation)."""
    mode = cfg.data.get("image_resize", "pad")
    if mode not in ("pad", "crop"):
        raise ValueError(f"data.image_resize không hợp lệ: {mode!r}, chọn pad | crop")
    steps = [PadToSquare(fill)] if mode == "pad" else []
    augment = build_image_augment(cfg.data.get("image_augment", False)) if train else None
    if augment is not None:
        steps.append(augment)
    return transforms.Compose(steps) if steps else None


def build_datasets(cfg, needs_text=True, needs_image=True, splits=("train", "val", "test")):
    records = load_all_records(cfg)
    text_pre = TextPreprocessor.from_config(cfg.data.text) if needs_text else None
    # màu viền khi pad và màu ảnh thay thế ảnh lỗi: mean của encoder, sau normalize gần 0
    fill = encoder_mean_color(cfg.model.image_encoder.name) if needs_image else (0, 0, 0)
    datasets = {}
    for split in splits:
        if split not in records:
            continue
        datasets[split] = ViMMSDDataset(
            records[split],
            text_preprocessor=text_pre,
            image_transform=build_image_transform(cfg, split == "train", fill) if needs_image else None,
            load_image=needs_image,
            cache_dir=text_cache_dir(cfg),
            use_image_text=uses_image_text(cfg) and needs_text,
            missing_image_color=fill,
        )
    return datasets


def class_counts(dataset, num_classes):
    counts = Counter(dataset.labels)
    return [counts.get(i, 0) for i in range(num_classes)]


def build_train_sampler(dataset, num_classes, mode):
    if mode == "random":
        return None
    if mode == "weighted":
        # oversampling: mỗi lớp được lấy mẫu với xác suất ngang nhau
        counts = class_counts(dataset, num_classes)
        weights = [1.0 / counts[l] for l in dataset.labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    raise ValueError(f"sampler không hợp lệ: {mode}")


def build_dataloaders(cfg, datasets, collator):
    num_classes = len(cfg.data.labels)
    loaders = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        sampler = build_train_sampler(ds, num_classes, cfg.data.sampler) if is_train else None
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.train.batch_size if is_train else cfg.train.eval_batch_size,
            shuffle=is_train and sampler is None,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
