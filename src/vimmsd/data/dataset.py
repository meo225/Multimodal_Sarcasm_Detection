import json
import logging
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vimmsd.data.augmentation import build_image_augment
from vimmsd.data.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

IGNORE_LABEL = -1  # mẫu không có nhãn (public test)


def load_records(json_path, image_dir, label2id, ocr_texts=None):
    """Đọc file annotation ViMMSD: {id: {"image", "caption", "label"}} (hoặc list các dict).
    Trả về list dict thống nhất: id, image_path, caption, label (int, -1 nếu không có nhãn)."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.items() if isinstance(raw, dict) else ((str(i), r) for i, r in enumerate(raw))

    records = []
    for sid, r in items:
        label = r.get("label")
        rec = {
            "id": str(sid),
            "image_path": str(Path(image_dir) / r["image"]),
            "caption": r.get("caption", ""),
            "label": label2id[label] if label is not None else IGNORE_LABEL,
        }
        if ocr_texts is not None:
            rec["ocr"] = ocr_texts.get(ocr_key(image_dir, r["image"]), "")
        records.append(rec)
    return records


def ocr_key(image_dir, image_name):
    # key không phụ thuộc đường dẫn tuyệt đối, để cache OCR tạo trên Kaggle dùng được ở local/Colab
    return f"{Path(image_dir).name}/{image_name}"


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
                 load_image=True, cache_dir=None, use_ocr=False):
        self.records = records
        self.image_transform = image_transform
        self.load_image = load_image
        self.use_ocr = use_ocr

        captions = [r["caption"] for r in records]
        if text_preprocessor is not None:
            captions = text_preprocessor.process_all(captions, cache_dir=cache_dir)
            if use_ocr:
                ocr = text_preprocessor.process_all([r.get("ocr", "") for r in records], cache_dir=cache_dir)
        elif use_ocr:
            ocr = [r.get("ocr", "") for r in records]
        self.texts = captions
        self.ocr_texts = ocr if use_ocr else None
        self._bad_images = 0

    def __len__(self):
        return len(self.records)

    @property
    def labels(self):
        return [r["label"] for r in self.records]

    def _open_image(self, path):
        try:
            img = Image.open(path)
            img.seek(0)  # ảnh GIF: lấy frame đầu
            return img.convert("RGB")
        except (OSError, ValueError) as e:
            self._bad_images += 1
            if self._bad_images <= 5:
                logger.warning("không đọc được ảnh %s (%s), thay bằng ảnh đen", path, e)
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx):
        rec = self.records[idx]
        item = {"id": rec["id"], "text": self.texts[idx], "label": rec["label"]}
        if self.ocr_texts is not None:
            item["ocr"] = self.ocr_texts[idx]
        if self.load_image:
            img = self._open_image(rec["image_path"])
            if self.image_transform is not None:
                img = self.image_transform(img)
            item["image"] = img
        return item


class ViMMSDCollator:
    """Tokenize text + xử lý ảnh theo batch (padding động theo câu dài nhất trong batch)."""

    def __init__(self, tokenizer=None, image_processor=None, max_length=128):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __call__(self, items):
        batch = {
            "ids": [it["id"] for it in items],
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
        }
        if self.tokenizer is not None:
            texts = [it["text"] for it in items]
            if "ocr" in items[0]:
                # caption và OCR là 2 segment: <s> caption </s></s> ocr </s>
                enc = self.tokenizer(texts, [it["ocr"] for it in items], padding=True,
                                     truncation="longest_first", max_length=self.max_length,
                                     return_tensors="pt")
            else:
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            batch["input_ids"] = enc["input_ids"]
            batch["attention_mask"] = enc["attention_mask"]
        if self.image_processor is not None:
            images = [it["image"] for it in items]
            batch["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return batch


def load_ocr_texts(cfg):
    if not cfg.data.get("use_ocr"):
        return None
    path = Path(cfg.paths.cache_dir) / cfg.data.ocr_cache
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_records(cfg):
    """Trả về dict split -> records: train/val/test (chia từ tập có nhãn) và public_test (không nhãn)."""
    data_dir = Path(cfg.paths.data_dir)
    label2id = {l: i for i, l in enumerate(cfg.data.labels)}
    ocr_texts = load_ocr_texts(cfg)

    labeled = load_records(data_dir / cfg.data.train_json, data_dir / cfg.data.train_image_dir,
                           label2id, ocr_texts)
    train, val, test = split_records(labeled, cfg.data.val_ratio, cfg.data.test_ratio, cfg.seed)
    splits = {"train": train, "val": val, "test": test}

    public_json = cfg.data.get("public_test_json")
    if public_json and (data_dir / public_json).exists():
        splits["public_test"] = load_records(data_dir / public_json,
                                             data_dir / cfg.data.public_test_image_dir,
                                             label2id, ocr_texts)
    return splits


def build_datasets(cfg, needs_text=True, needs_image=True, splits=("train", "val", "test")):
    records = load_all_records(cfg)
    text_pre = TextPreprocessor.from_config(cfg.data.text) if needs_text else None
    datasets = {}
    for split in splits:
        if split not in records:
            continue
        augment = build_image_augment(cfg.data.get("image_augment", False)) if split == "train" else None
        datasets[split] = ViMMSDDataset(
            records[split],
            text_preprocessor=text_pre,
            image_transform=augment,
            load_image=needs_image,
            cache_dir=cfg.paths.get("cache_dir"),
            use_ocr=bool(cfg.data.get("use_ocr")) and needs_text,
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
