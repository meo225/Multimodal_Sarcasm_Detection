"""Test nhanh không cần mạng/GPU: dữ liệu giả + tokenizer/image processor giả, kiểm tra shape và encoding."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from vimmsd.analysis.shortcut_check import derangement, swap_images
from vimmsd.data.dataset import ViMMSDCollator, ViMMSDDataset, load_records, split_records
from vimmsd.data.preprocessing import TextPreprocessor, clean_ocr, clean_text
from vimmsd.models.fusion import FUSIONS
from vimmsd.training.losses import FocalLoss, build_loss, compute_class_weights
from vimmsd.training.metrics import compute_metrics
from vimmsd.utils.config import Config, load_config

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]


@pytest.fixture
def fake_data(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ann = {}
    for i in range(40):
        name = f"{i}.jpg"
        Image.new("RGB", (64 + i, 48), color=(i * 5, 0, 0)).save(img_dir / name)
        ann[str(i)] = {"image": name, "caption": f"Ảnh số {i} ko đẹp lắm 😂 #meme", "label": LABELS[i % 4]}
    (tmp_path / "train.json").write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    return tmp_path


class StubTokenizer:
    def __call__(self, texts, pairs=None, padding=True, truncation=True, max_length=16, return_tensors="pt", **_):
        lengths = [min(len(t.split()), max_length) for t in texts]
        L = max(lengths)
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, n in enumerate(lengths):
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2


class StubImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.stack([torch.zeros(3, 32, 32) for _ in images])}


def test_clean_text_keeps_vietnamese_and_numbers():
    out = clean_text("Giá 1000đ mà quáaaaa đắt, ko mua đc https://x.com @ban",
                     emoji_mode="keep", word_segment_=False)
    assert out == "Giá 1000đ mà quá đắt, không mua được"


def test_clean_text_normalizes_unicode():
    decomposed = "hoà"  # "hoà" viết bằng dấu tổ hợp
    assert clean_text(decomposed, emoji_mode="keep", word_segment_=False) == "hoà"


def test_clean_text_unescapes_html_and_teencode():
    text = "Mn ơi cmt này đỉnh thui &amp; nhìn mlem ghê á, ko mua đc"
    out = clean_text(text, emoji_mode="keep", word_segment_=False)
    assert "mọi người" in out and "bình luận" in out and "&" in out and "không" in out and "được" in out


def test_clean_text_preserves_sensitive_words():
    # Các từ có nguy cơ nhầm lẫn nhưng phải giữ nguyên vẹn
    text = "Ăn sữa chua cay, nặng 70 kg, sinh SN 1999, mua bảo hiểm Dr.G và biết PK game"
    out = clean_text(text, emoji_mode="keep", word_segment_=False)
    assert "sữa chua cay" in out
    assert "70 kg" in out
    assert "SN 1999" in out
    assert "bảo hiểm" in out
    assert "Dr.G" in out
    assert "PK game" in out


def test_clean_text_teencode_with_punctuation_and_caps():
    # Teencode viết hoa hoặc dính dấu câu
    text = "Khum? ĐC! (Ko) \"ntn\"... wa' đã"
    out = clean_text(text, emoji_mode="keep", word_segment_=False)
    assert "không?" in out.lower()
    assert "được!" in out.lower()
    assert "(không)" in out.lower()
    assert "\"như thế nào\"" in out.lower()


def test_clean_text_emoji_modes():
    text = "Đỉnh quá 😂 🐧"
    assert clean_text(text, emoji_mode="demojize", word_segment_=False) == "Đỉnh quá cười ra nước mắt chim cánh cụt"
    assert "😂" in clean_text(text, emoji_mode="keep", word_segment_=False)
    assert "😂" not in clean_text(text, emoji_mode="remove", word_segment_=False)


def test_clean_text_emoji_sequences_stay_whole():
    # emoji ghép bằng ZWJ và có màu da: xử lý như một emoji, không tách thành "person ... male sign"
    out = clean_text("chịu 🤦🏻\u200d♂️ 👍🏽 ❤️\u200d🔥 ‼️", word_segment_=False)
    assert out == "chịu man facepalming light skin tone ngón cái giơ lên trái tim rực lửa !!"


def test_clean_text_invisible_chars_do_not_split_words():
    assert clean_text("kh\u200bông th\u200cích a\xa0b", emoji_mode="keep", word_segment_=False) == "không thích a b"


def test_collapse_repeats_keeps_acronyms():
    out = clean_text("PCCC CCCD VIII quáaaa đẹppppp Duaaa ĐẸPPPP", emoji_mode="keep", word_segment_=False)
    assert out == "PCCC CCCD VIII quá đẹp Dua ĐẸPPPP"


def test_teencode_skips_ambiguous_words():
    text = "Gen Z, K-Pop, súng AK, du học UK, lỗ hổng, tối thui, bít tắc, KO MUA ĐC, ko bít j z"
    out = clean_text(text, emoji_mode="keep", word_segment_=False)
    assert out == "Gen Z, K-Pop, súng AK, du học UK, lỗ hổng, tối thui, bít tắc, không MUA được, không bít gì vậy"


def test_extract_emojis_keeps_order_repeats_and_sequences():
    from vimmsd.data.preprocessing import extract_emojis

    assert extract_emojis("Đỉnh 😂😂 quá 🤦🏻\u200d♂️!") == ["😂", "😂", "🤦🏻\u200d♂️"]
    assert extract_emojis(None) == []


def test_clean_ocr_drops_junk_lines_and_caption_duplicates():
    ocr = "Tyler, The Creator\n8\n@ty\nLàm hộ chiếu đi\n0889 24 24\n:"
    assert clean_ocr(ocr, "caption khác") == "Tyler, The Creator Làm hộ chiếu đi"
    assert clean_ocr("Ước gì nằm yên cũng được nhiều like", "Ước gì nằm yên cũng được nhiều like!") == ""
    assert clean_ocr("", "abc") == ""


def test_text_cache_file_has_fixed_name(tmp_path):
    pre = TextPreprocessor(emoji="keep", word_segment=False)
    assert pre._cache_file(tmp_path).name == "01a_text_preprocessing.json"


def test_clean_text_empty_and_whitespace():
    assert clean_text("", word_segment_=False) == ""
    assert clean_text("   \n\t   ", word_segment_=False) == ""
    assert clean_text(None, word_segment_=False) == ""


def test_clean_text_word_segment_underthesea():
    text = "Học sinh sinh viên dùng mạng xã hội"
    out = clean_text(text, emoji_mode="keep", word_segment_="underthesea")
    assert "học_sinh" in out.lower() or "sinh_viên" in out.lower() or "mạng_xã_hội" in out.lower()


def test_clean_text_word_segment_vncorenlp():
    # cần Java và model VnCoreNLP đã tải về (lần đầu chạy pipeline sẽ tự tải), không có thì bỏ qua
    import shutil

    from vimmsd.data.preprocessing import VNCORENLP_DIR, VNCORENLP_FILES

    pytest.importorskip("py_vncorenlp")
    if not shutil.which("java") or not all((VNCORENLP_DIR / f).exists() for f in VNCORENLP_FILES):
        pytest.skip("thiếu Java hoặc model VnCoreNLP")
    out = clean_text("Học sinh sinh viên dùng mạng xã hội", emoji_mode="keep", word_segment_="vncorenlp")
    assert out == "Học_sinh sinh_viên dùng mạng xã_hội"


def test_text_aux_fusion_keeps_caption_as_main():
    from vimmsd.models.fusion import TextAuxFusion

    fusion = TextAuxFusion(4, ocr_weight=0.5, description_weight=0.25)
    caption = torch.ones(2, 4)
    ocr = torch.full((2, 4), 10.0)
    description = torch.full((2, 4), 5.0)
    present = torch.tensor([1.0, 0.0])
    out = fusion({
        "pooled": caption,
        "ocr": ocr,
        "ocr_present": present,
        "description": description,
        "description_present": present,
    })
    assert fusion.weights() == {"caption": 1.0, "ocr": 0.5, "description": 0.25}
    assert torch.allclose(out[0], caption[0] + 0.5 * ocr[0] + 0.25 * description[0])
    assert torch.allclose(out[1], caption[1])
    with pytest.raises(ValueError):
        TextAuxFusion(4, ocr_weight=0.2, description_weight=0.5)


def test_text_aux_collator_does_not_merge_fields():
    items = [
        {"id": "0", "label": 0, "text": "caption một", "ocr": "chữ OCR", "description": ""},
        {"id": "1", "label": 1, "text": "caption hai", "ocr": "", "description": "một mô tả"},
    ]
    batch = ViMMSDCollator(StubTokenizer(), max_length=16, aux_max_length=8)(items)
    assert "input_ids" in batch and "ocr_input_ids" in batch and "description_input_ids" in batch
    assert batch["ocr_present"].tolist() == [1.0, 0.0]
    assert batch["description_present"].tolist() == [0.0, 1.0]
    assert batch["ocr_input_ids"].shape[0] == 2


def test_collator_clips_caption_before_only_second():
    # caption dài hơn max_length: cắt caption trước để only_second không trả về chuỗi quá dài
    seen = {}

    class PairTokenizer(StubTokenizer):
        def __call__(self, texts, pairs=None, **kw):
            seen["texts"], seen["kw"] = texts, kw
            return super().__call__(texts, **kw)

    items = [{"id": "0", "label": 0, "text": " ".join(["w"] * 50), "image_text": "chữ trong ảnh"}]
    ViMMSDCollator(PairTokenizer(), max_length=20, max_caption_length=200)(items)
    assert len(seen["texts"][0].split()) == 20 - 4 - 1 and seen["kw"]["truncation"] == "only_second"
    ViMMSDCollator(PairTokenizer(), max_length=20, max_caption_length=10)(items)
    assert len(seen["texts"][0].split()) == 10



def test_load_and_split(fake_data):
    records = load_records(fake_data / "train.json", fake_data / "images", {l: i for i, l in enumerate(LABELS)})
    assert len(records) == 40 and {r["label"] for r in records} == {0, 1, 2, 3}
    train, val, test = split_records(records, 0.2, 0.2, seed=0)
    assert len(train) + len(val) + len(test) == 40
    assert {r["label"] for r in test} == {0, 1, 2, 3}  # stratified
    assert not {r["id"] for r in train} & {r["id"] for r in test}


def _write_test_split(root, name, n, offset):
    ann = {str(i): {"image": f"{i}.jpg", "caption": f"mẫu {name} {i}", "label": None} for i in range(n)}
    for i in range(n):
        Image.new("RGB", (32, 32)).save(root / "images" / f"{i}.jpg")
    (root / f"{name}.json").write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    ref = {"results": {str(i): LABELS[(i + offset) % 4] for i in range(n)}}
    (root / f"{name}-ref.json").write_text(json.dumps(ref), encoding="utf-8")


def _split_cfg(root, *extra):
    from vimmsd.data.dataset import load_all_records

    cfg = load_config("configs/base.yaml", overrides=[
        f"paths.local.data_dir={root}",
        "data.train_json=train.json", "data.train_image_dir=images",
        "data.public_test_json=public.json", "data.public_test_ref=public-ref.json",
        "data.public_test_image_dir=images",
        "data.private_test_json=private.json", "data.private_test_ref=private-ref.json",
        "data.private_test_image_dir=images",
        *extra,
    ], env="local")
    return load_all_records(cfg)


def test_competition_split_uses_ref_labels(fake_data):
    _write_test_split(fake_data, "public", 8, 0)
    _write_test_split(fake_data, "private", 12, 1)
    splits = _split_cfg(fake_data, "data.split_mode=competition")
    assert [len(splits[k]) for k in ("train", "val", "test")] == [40, 8, 12]
    assert [r["label"] for r in splits["val"]] == [i % 4 for i in range(8)]
    assert [r["label"] for r in splits["test"]] == [(i + 1) % 4 for i in range(12)]
    assert "public_test" not in splits


def test_competition_split_rejects_missing_ref_label(fake_data):
    _write_test_split(fake_data, "public", 8, 0)
    _write_test_split(fake_data, "private", 12, 1)
    ref = json.loads((fake_data / "private-ref.json").read_text())
    del ref["results"]["3"]
    (fake_data / "private-ref.json").write_text(json.dumps(ref), encoding="utf-8")
    with pytest.raises(ValueError, match="private_test"):
        _split_cfg(fake_data, "data.split_mode=competition")


def test_random_split_keeps_public_test_unlabeled(fake_data):
    _write_test_split(fake_data, "public", 8, 0)
    _write_test_split(fake_data, "private", 12, 1)
    splits = _split_cfg(fake_data, "data.split_mode=random")
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 40
    assert {r["label"] for r in splits["public_test"]} == {-1}


def test_from_config_ignores_cache_dir():
    cfg = load_config("configs/base.yaml")
    pre = TextPreprocessor.from_config(cfg.data.text)
    assert pre.kwargs["emoji_mode"] == "demojize"
    pre = TextPreprocessor.from_config({
        "emoji": "keep", "word_segment": False, "cache_dir": "attached/cache",
    })
    assert pre.kwargs["emoji_mode"] == "keep"
    assert "cache_dir" not in pre.kwargs


def test_build_text_cache_is_reused(fake_data):
    from vimmsd.data.dataset import build_text_cache, text_cache_dir

    cache = fake_data / "cache"
    cfg = load_config("configs/base.yaml", overrides=[
        f"paths.local.data_dir={fake_data}",
        f"paths.local.cache_dir={cache}",
        "data.train_json=train.json",
        "data.public_test_json=missing-public.json",
        "data.private_test_json=missing-private.json",
        "data.text.word_segment=false",
        "data.text.emoji=keep",
    ], env="local")
    assert text_cache_dir(cfg) == str(cache)
    cache_file, n_captions, used = build_text_cache(cfg)
    assert n_captions == 40 and used == ["train.json"] and cache_file.exists()

    records = load_records(fake_data / "train.json", fake_data / "images", {l: i for i, l in enumerate(LABELS)})
    ds = ViMMSDDataset(records, text_preprocessor=TextPreprocessor.from_config(cfg.data.text), cache_dir=cache)
    assert "không" in ds.texts[0]
    # lần sau chỉ đọc file, không ghi thêm key mới
    before = cache_file.read_text(encoding="utf-8")
    TextPreprocessor.from_config(cfg.data.text).process_all([r["caption"] for r in records], cache_dir=cache)
    assert cache_file.read_text(encoding="utf-8") == before


def test_stage_preprocessed_cache_copies_json_not_csv(tmp_path):
    from vimmsd.data.dataset import stage_preprocessed_cache

    src = tmp_path / "input" / "vimmsd-data-preprocessed"
    src.mkdir(parents=True)
    (src / "text_preprocessed.json").write_text('{"texts": {}}', encoding="utf-8")
    (src / "image_preprocessed.json").write_text('{"images": {}}', encoding="utf-8")
    (src / "image_preprocessed.csv").write_text("split,id\n", encoding="utf-8")
    dest = tmp_path / "cache"
    cfg = Config(
        paths=Config(cache_dir=str(dest)),
        data=Config(
            text=Config(cache_file="text_preprocessed.json"),
            image_text=Config(use_ocr=True, use_description=True, file="image_preprocessed.json"),
        ),
    )
    copied = stage_preprocessed_cache(cfg, search_root=tmp_path / "input")
    assert [p.name for p in copied] == ["text_preprocessed.json", "image_preprocessed.json"]
    assert not (dest / "image_preprocessed.csv").exists()


def test_process_all_reads_kaggle_text_cache(tmp_path):
    pre = TextPreprocessor(emoji="keep", word_segment=False, cache_file="text_preprocessed.json")
    path = tmp_path / "text_preprocessed.json"
    path.write_text(json.dumps({
        "meta": {"pipeline_key": pre.pipeline_key(), "preprocess_version": 2},
        "texts": {"ko đẹp": "ĐÃ CACHE"},
    }, ensure_ascii=False), encoding="utf-8")
    assert pre.process_all(["ko đẹp", "ko đẹp"], cache_dir=tmp_path) == ["ĐÃ CACHE", "ĐÃ CACHE"]
    # cache đủ thì không ghi lại
    assert "ĐÃ CACHE" in path.read_text(encoding="utf-8")


def test_process_all_ignores_cache_when_preprocess_version_differs(tmp_path):
    pre = TextPreprocessor(emoji="keep", word_segment=False, cache_file="text_preprocessed.json")
    (tmp_path / "text_preprocessed.json").write_text(json.dumps({
        "meta": {"preprocess_version": 1},
        "texts": {"ko": "ĐÃ CACHE"},
    }), encoding="utf-8")
    assert pre.process_all(["ko"], cache_dir=tmp_path) == ["không"]


def test_from_config_uses_cache_file_name(tmp_path):
    pre = TextPreprocessor.from_config({
        "emoji": "keep", "word_segment": False, "cache_file": "text_preprocessed.json", "cache_dir": "ignored",
    })
    assert pre._cache_file(tmp_path).name == "text_preprocessed.json"
    assert "cache_dir" not in pre.kwargs


def test_load_combined_image_preprocessed(fake_data):
    from vimmsd.data.dataset import load_image_texts

    cache = fake_data / "cache"
    cache.mkdir()
    (cache / "image_preprocessed.json").write_text(json.dumps({
        "meta": {"num_images": 1, "missing": {"ocr": 0, "description": 0}},
        "images": {"images/0.jpg": {"ocr": "chữ trên ảnh", "description": "một người"},
                   "images/1.jpg": {"ocr": "", "description": None}},
    }, ensure_ascii=False), encoding="utf-8")
    cfg = Config(paths=Config(cache_dir=str(cache)), data=Config(image_text=Config(
        use_ocr=True, use_description=True, dir=None, file="image_preprocessed.json")))
    texts = load_image_texts(cfg)
    assert texts["ocr"]["images/0.jpg"] == "chữ trên ảnh"
    assert texts["ocr"]["images/1.jpg"] == "" and "images/1.jpg" not in texts["description"]  # null = chưa chạy

    records = load_records(fake_data / "train.json", fake_data / "images",
                           {l: i for i, l in enumerate(LABELS)}, texts)
    ds = ViMMSDDataset(records[:2], text_preprocessor=TextPreprocessor(emoji="keep", word_segment=False),
                       load_image=False, use_image_text=True, cache_dir=cache)
    assert ds[0]["image_text"] == "Chữ trong ảnh: chữ trên ảnh. Mô tả ảnh: một người"
    assert ds[1]["image_text"] == ""


def test_batch_shapes(fake_data):
    records = load_records(fake_data / "train.json", fake_data / "images", {l: i for i, l in enumerate(LABELS)})
    pre = TextPreprocessor(emoji="keep", word_segment=False)
    ds = ViMMSDDataset(records, text_preprocessor=pre, cache_dir=fake_data / "cache")
    assert "không" in ds.texts[0] and "#" not in ds.texts[0]
    assert (fake_data / "cache" / "01a_text_preprocessing.json").exists()

    batch = ViMMSDCollator(StubTokenizer(), StubImageProcessor())([ds[i] for i in range(8)])
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["pixel_values"].shape == (8, 3, 32, 32)
    assert batch["labels"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


@pytest.mark.parametrize("modality", list(FUSIONS))
def test_fusion_shapes(modality):
    B, Lt, Li, Ht, Hi = 3, 7, 50, 768, 768
    mask = torch.ones(B, Lt, dtype=torch.bool)
    mask[0, 4:] = False
    text = {"tokens": torch.randn(B, Lt, Ht), "mask": mask, "pooled": torch.randn(B, Ht)}
    image = {"tokens": torch.randn(B, Li, Hi), "pooled": torch.randn(B, Hi)}
    fusion = FUSIONS[modality](text_dim=Ht, image_dim=Hi, hidden_dim=64, num_heads=4, dropout=0.0)
    out = fusion(text, image)
    assert out.shape == (B, fusion.out_dim)
    if modality == "cross_attn":
        # patch ảnh không được attend vào token padding
        assert torch.allclose(fusion.last_attn["image_to_text"][0, :, 4:], torch.tensor(0.0))


def test_losses():
    counts = [6000, 80, 400, 4000]
    w = compute_class_weights(counts, "effective")
    assert w.argmax() == 1 and torch.isclose(w.mean(), torch.tensor(1.0))
    logits, y = torch.randn(10, 4), torch.randint(0, 4, (10,))
    # focal loss với gamma=0 và không có weight chính là cross entropy
    assert torch.isclose(FocalLoss(gamma=0.0)(logits, y), torch.nn.functional.cross_entropy(logits, y))
    loss = build_loss(Config(name="focal", gamma=2.0, class_weight="inverse"), counts)
    assert loss(logits, y).item() > 0


def test_metrics():
    m = compute_metrics([0, 1, 2, 3, 0], [0, 1, 2, 0, 0], LABELS)
    assert m["per_class"]["multi-sarcasm"]["recall"] == 0.0
    assert m["confusion_matrix"][3][0] == 1


def test_config_inherit_override_and_env(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text(
        "seed: 1\npaths:\n  local: {data_dir: data/raw}\n  kaggle: {data_dir: /kaggle/input/x}\n"
        "train: {lr: 0.1, epochs: 3}\n"
    )
    (cfg_dir / "exp.yaml").write_text("inherit: base.yaml\ntrain: {epochs: 5}\n")
    cfg = load_config(cfg_dir / "exp.yaml", ["train.lr=1e-5"], env="local")
    assert cfg.train.epochs == 5 and cfg.train.lr == 1e-5 and cfg.seed == 1
    assert cfg.paths.data_dir == str(tmp_path / "data/raw") and cfg.name == "exp"
    assert load_config(cfg_dir / "exp.yaml", env="kaggle").paths.data_dir == "/kaggle/input/x"


def test_image_swap_keeps_text():
    rng = np.random.default_rng(0)
    perm = derangement(10, rng)
    assert not np.any(perm == np.arange(10))
    records = [{"id": str(i), "caption": f"c{i}", "image_path": f"{i}.jpg", "label": 3} for i in range(10)]
    swapped = swap_images(records, seed=0)
    assert [r["caption"] for r in swapped] == [r["caption"] for r in records]
    assert all(a["image_path"] != b["image_path"] for a, b in zip(records, swapped))


def test_image_text_cache_resume_and_compose(tmp_path):
    from vimmsd.data.image_text import build_image_text_cache, compose_image_text, list_images

    img_dir = tmp_path / "train-images"
    img_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8)).save(img_dir / f"{i}.jpg")
    calls = []

    def fake(path):
        calls.append(path.name)
        if path.name == "2.jpg":
            raise RuntimeError("ảnh lỗi")
        return f"text {path.stem}"

    out = tmp_path / "cache" / "ocr.json"
    cache = build_image_text_cache(list_images([img_dir]), out, fake, save_every=1)
    # ảnh lỗi không được ghi vào cache: chuỗi rỗng chỉ dành cho ảnh không có text
    assert cache == {"train-images/0.jpg": "text 0", "train-images/1.jpg": "text 1"}
    assert json.loads(out.read_text(encoding="utf-8")) == cache
    calls.clear()
    build_image_text_cache(list_images([img_dir]), out, fake)
    assert calls == ["2.jpg"]  # chạy lại: bỏ qua ảnh đã có trong cache, thử lại ảnh lỗi

    assert compose_image_text("KHI MÀI", "Một con bò") == "Chữ trong ảnh: KHI MÀI. Mô tả ảnh: Một con bò"
    assert compose_image_text("", "Một con bò") == "Mô tả ảnh: Một con bò"
    assert compose_image_text(" ", "") == ""


def test_image_text_cache_stops_when_every_image_fails(tmp_path):
    from vimmsd.data.image_text import build_image_text_cache, list_images

    img_dir = tmp_path / "train-images"
    img_dir.mkdir()
    for i in range(6):
        Image.new("RGB", (8, 8)).save(img_dir / f"{i}.jpg")
    calls = []

    def broken(path):
        calls.append(path.name)
        raise RuntimeError("hết VRAM")

    out = tmp_path / "cache" / "ocr.json"
    with pytest.raises(RuntimeError, match="3 ảnh lỗi liên tiếp"):
        build_image_text_cache(list_images([img_dir]), out, broken, max_consecutive_failures=3)
    assert len(calls) == 3 and json.loads(out.read_text(encoding="utf-8")) == {}


def test_reading_order():
    from vimmsd.data.image_text import reading_order

    # 2 dòng, dòng trên có 2 box lệch nhau vài pixel theo chiều dọc
    right, left, below = (60, 12, 100, 32), (0, 10, 50, 30), (0, 40, 100, 60)
    assert reading_order([below, right, left]) == [left, right, below]
    assert reading_order([]) == []


def test_dataset_with_image_text(fake_data):
    from vimmsd.data.dataset import load_image_texts

    (fake_data / "cache").mkdir(exist_ok=True)
    (fake_data / "cache" / "ocr.json").write_text(json.dumps({"images/0.jpg": "chữ trên ảnh"}), encoding="utf-8")
    (fake_data / "cache" / "desc.json").write_text(json.dumps({"images/0.jpg": "một người"}), encoding="utf-8")
    cfg = Config(paths=Config(cache_dir=str(fake_data / "cache")), data=Config(image_text=Config(
        use_ocr=True, use_description=True, dir=None, ocr_cache="ocr.json", description_cache="desc.json")))
    texts = load_image_texts(cfg)

    records = load_records(fake_data / "train.json", fake_data / "images",
                           {l: i for i, l in enumerate(LABELS)}, texts)
    assert records[0]["ocr"] == "chữ trên ảnh" and records[1]["description"] == ""
    ds = ViMMSDDataset(records, text_preprocessor=TextPreprocessor(emoji="keep", word_segment=False),
                       load_image=False, use_image_text=True)
    assert ds[0]["image_text"] == "Chữ trong ảnh: chữ trên ảnh. Mô tả ảnh: một người"
    assert ds[1]["image_text"] == ""

    seen = {}

    class PairTokenizer(StubTokenizer):
        def __call__(self, texts, pairs=None, **kw):
            seen["pairs"] = pairs
            return super().__call__(texts, **kw)

    ViMMSDCollator(PairTokenizer())([ds[0], ds[1]])
    assert seen["pairs"][0].startswith("Chữ trong ảnh")

    cfg.data.image_text.description_cache = "missing.json"
    with pytest.raises(FileNotFoundError):
        load_image_texts(cfg)

    # cache không khớp ảnh nào (sai thư mục, chưa chạy split này) phải báo lỗi thay vì âm thầm trả text rỗng
    with pytest.raises(ValueError, match="không có ảnh nào"):
        load_records(fake_data / "train.json", fake_data / "images", {l: i for i, l in enumerate(LABELS)},
                     {"ocr": {"other-images/0.jpg": "x"}})


def test_open_image_rgb_handles_gif_exif_and_transparency(tmp_path):
    from vimmsd.data.image_io import open_image_rgb

    # GIF nhiều frame: lấy frame đầu
    frames = [Image.new("RGB", (8, 8), c) for c in [(255, 0, 0), (0, 0, 255)]]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:])
    img = open_image_rgb(tmp_path / "a.gif")
    assert img.mode == "RGB" and img.getpixel((0, 0))[0] > 200

    # EXIF orientation = 6 (xoay 90 độ): ảnh 20x10 phải thành 10x20
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (20, 10)).save(tmp_path / "b.jpg", exif=exif)
    assert open_image_rgb(tmp_path / "b.jpg").size == (10, 20)

    # vùng trong suốt thành nền trắng, không thành đen
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgba.putpixel((0, 0), (0, 0, 0, 255))
    rgba.save(tmp_path / "c.png")
    img = open_image_rgb(tmp_path / "c.png")
    assert img.getpixel((3, 3)) == (255, 255, 255) and img.getpixel((0, 0)) == (0, 0, 0)


def test_pad_to_square_keeps_whole_image():
    from vimmsd.data.image_io import PadToSquare

    img = Image.new("RGB", (30, 90), (255, 255, 255))
    out = PadToSquare((122, 116, 104))(img)
    assert out.size == (90, 90)
    assert out.getpixel((0, 0)) == (122, 116, 104) and out.getpixel((45, 45)) == (255, 255, 255)
    assert PadToSquare()(Image.new("RGB", (5, 5))).size == (5, 5)


def test_build_image_transform_modes():
    from vimmsd.data.dataset import build_image_transform

    cfg = Config(data=Config(image_resize="pad", image_augment=True))
    tall = Image.new("RGB", (40, 120))
    assert build_image_transform(cfg, train=False, fill=(0, 0, 0))(tall).size == (120, 120)
    assert build_image_transform(cfg, train=True, fill=(0, 0, 0))(tall).size == (224, 224)
    cfg.data.image_resize, cfg.data.image_augment = "crop", False
    assert build_image_transform(cfg, train=True, fill=(0, 0, 0)) is None
    cfg.data.image_resize = "stretch"
    with pytest.raises(ValueError):
        build_image_transform(cfg, train=False, fill=(0, 0, 0))


def test_broken_image_is_flagged(fake_data):
    records = load_records(fake_data / "train.json", fake_data / "images", {l: i for i, l in enumerate(LABELS)})
    (fake_data / "images" / "1.jpg").write_bytes(b"not an image")
    ds = ViMMSDDataset(records[:2], missing_image_color=(1, 2, 3))
    assert ds[0]["img_missing"] == 0
    assert ds[1]["img_missing"] == 1 and ds[1]["image"].getpixel((0, 0)) == (1, 2, 3)
    batch = ViMMSDCollator(None, StubImageProcessor())([ds[0], ds[1]])
    assert batch["img_missing"].tolist() == [0, 1]


def test_image_text_cache_batches_and_isolates_failures(tmp_path):
    from vimmsd.data.image_text import build_image_text_cache, list_images

    img_dir = tmp_path / "train-images"
    img_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (8, 8)).save(img_dir / f"{i}.jpg")

    class FakeOCR:
        def __init__(self):
            self.batches = []

        def extract_many(self, paths):
            self.batches.append([p.name for p in paths])
            if any(p.name == "3.jpg" for p in paths):
                raise RuntimeError("ảnh lỗi trong nhóm")
            return [f"text {p.stem}" for p in paths]

        def __call__(self, path):
            if path.name == "3.jpg":
                raise RuntimeError("ảnh lỗi")
            return f"text {path.stem}"

    ocr = FakeOCR()
    out = tmp_path / "cache" / "ocr.json"
    cache = build_image_text_cache(list_images([img_dir]), out, ocr, batch_size=2, save_every=1)
    assert ocr.batches == [["0.jpg", "1.jpg"], ["2.jpg", "3.jpg"], ["4.jpg"]]
    # nhóm có ảnh lỗi được chạy lại từng ảnh: chỉ ảnh lỗi bị bỏ qua
    assert cache == {f"train-images/{i}.jpg": f"text {i}" for i in (0, 1, 2, 4)}


def test_shard_cache_merge(tmp_path):
    from vimmsd.data.image_text import merge_image_text_caches, shard_cache_name

    assert shard_cache_name("ocr.json", 1, 2) == "ocr.shard1of2.json"
    (tmp_path / "ocr.json").write_text(json.dumps({"a/0.jpg": "cũ"}), encoding="utf-8")
    for k, data in enumerate([{"a/1.jpg": "x"}, {"a/2.jpg": "y"}]):
        (tmp_path / shard_cache_name("ocr.json", k, 2)).write_text(json.dumps(data), encoding="utf-8")
    merged = merge_image_text_caches(tmp_path / "ocr.json", sorted(tmp_path.glob("ocr.shard*.json")))
    assert merged == {"a/0.jpg": "cũ", "a/1.jpg": "x", "a/2.jpg": "y"}


def test_merged_image_text_file(fake_data):
    from vimmsd.data.dataset import load_image_texts
    from vimmsd.data.image_text import merge_image_texts

    images = merge_image_texts({"images/0.jpg": "chữ", "images/1.jpg": ""}, {"images/0.jpg": "một người"},
                               keys=["images/0.jpg", "images/1.jpg", "images/2.jpg"])
    assert images["images/1.jpg"] == {"ocr": "", "description": None}  # "" = không có chữ, None = chưa chạy
    assert images["images/2.jpg"] == {"ocr": None, "description": None}

    (fake_data / "merged").mkdir()
    (fake_data / "merged" / "image_text.json").write_text(json.dumps({"meta": {}, "images": images}), encoding="utf-8")
    cfg = Config(paths=Config(cache_dir=str(fake_data / "cache")), data=Config(image_text=Config(
        use_ocr=True, use_description=True, dir=str(fake_data / "merged"), file="image_text.json",
        ocr_cache="ocr.json", description_cache="description.json")))
    texts = load_image_texts(cfg)
    assert texts == {"ocr": {"images/0.jpg": "chữ", "images/1.jpg": ""}, "description": {"images/0.jpg": "một người"}}
    records = load_records(fake_data / "train.json", fake_data / "images", {l: i for i, l in enumerate(LABELS)}, texts)
    assert records[0]["ocr"] == "chữ" and records[0]["description"] == "một người" and records[1]["description"] == ""

    cfg.data.image_text.file = "missing.json"
    with pytest.raises(FileNotFoundError):
        load_image_texts(cfg)


def test_named_text_cache_checks_pipeline_key(tmp_path):
    pre = TextPreprocessor(emoji="keep", word_segment=False, cache_file="text_preprocessed.json")
    out = pre.process_all(["ko mua đc"], cache_dir=tmp_path)
    data = json.loads((tmp_path / "text_preprocessed.json").read_text(encoding="utf-8"))
    assert out == ["không mua được"] and data["meta"]["pipeline_key"] == pre.pipeline_key()
    assert data["texts"] == {"ko mua đc": "không mua được"}

    # cache tạo bởi pipeline khác (key khác) hoặc không có meta thì bị bỏ qua, không dùng nhầm
    data["meta"]["pipeline_key"] = "khác"
    data["texts"]["ko mua đc"] = "kết quả cũ"
    (tmp_path / "text_preprocessed.json").write_text(json.dumps(data), encoding="utf-8")
    assert pre.process_all(["ko mua đc"], cache_dir=tmp_path) == ["không mua được"]
    (tmp_path / "text_preprocessed.json").write_text(json.dumps({"ko mua đc": "kết quả cũ"}), encoding="utf-8")
    assert pre.process_all(["ko mua đc"], cache_dir=tmp_path) == ["không mua được"]
