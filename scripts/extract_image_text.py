"""Tiền xử lý ảnh thành text, chạy offline một lần trước khi train (cần GPU cho VLM).

    python scripts/extract_image_text.py --config configs/base.yaml --task ocr
    python scripts/extract_image_text.py --config configs/base.yaml --task description --vlm vintern

Kết quả lưu vào <data.image_text.dir hoặc paths.cache_dir>/<ocr_cache | description_cache>.
Chạy song song trên 2 GPU (Kaggle T4 x2): 2 tiến trình với CUDA_VISIBLE_DEVICES=0/1 và --num-shards 2 --shard 0/1,
rồi gộp các file shard bằng `vimmsd.data.image_text.merge_image_text_caches` (xem notebooks/01b).
Chạy tiếp được nếu bị ngắt: ảnh đã có trong file cache sẽ được bỏ qua. Ảnh lỗi không được ghi vào cache,
script thoát với mã lỗi nếu còn ảnh thiếu; chạy lại để thử lại các ảnh đó."""
import argparse
import json
import logging
import sys
from pathlib import Path

from vimmsd.data.image_text import (OCRExtractor, VLMDescriber, build_image_text_cache, image_key, list_images,
                                    shard_cache_name)
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--task", required=True, choices=["ocr", "description"])
    parser.add_argument("--vlm", default="vintern", choices=["vintern", "hf"], help="backend VLM cho --task description")
    parser.add_argument("--model", default=None, help="tên model HF, mặc định theo backend")
    parser.add_argument("--splits", nargs="*", default=["train", "public_test", "private_test"])
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu mỗi split (để thử)")
    parser.add_argument("--ocr-model", default="vgg_seq2seq", choices=["vgg_seq2seq", "vgg_transformer"],
                        help="model VietOCR: vgg_seq2seq nhanh hơn nhiều, vgg_transformer chính xác hơn một chút")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="số ảnh xử lý mỗi lần (OCR: gộp dòng chữ để nhận dạng; VLM: sinh mô tả theo batch)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="chia ảnh thành N phần để chạy song song (mỗi GPU một tiến trình), mỗi phần một file cache")
    parser.add_argument("--shard", type=int, default=0, help="phần thứ mấy (0..N-1) mà tiến trình này chạy")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(args.config, args.override)
    data_dir = Path(cfg.paths.data_dir)
    image_dirs = {
        "train": cfg.data.train_image_dir,
        "public_test": cfg.data.get("public_test_image_dir"),
        "private_test": cfg.data.get("private_test_image_dir"),
    }
    paths = []
    for split in args.splits:
        if image_dirs.get(split) and (data_dir / image_dirs[split]).exists():
            images = list_images([data_dir / image_dirs[split]])
            paths.extend(images[: args.limit] if args.limit else images)
        else:
            logging.warning("bỏ qua split %s: không có thư mục ảnh", split)
    paths = paths[args.shard::args.num_shards]

    it_cfg = cfg.data.image_text
    out_dir = Path(it_cfg.get("dir") or cfg.paths.cache_dir)
    if args.task == "ocr":
        extractor, out_name = OCRExtractor(rec_model=args.ocr_model), it_cfg.ocr_cache
    else:
        extractor, out_name = VLMDescriber(backend=args.vlm, model_name=args.model), it_cfg.description_cache
    if args.num_shards > 1:
        # ảnh đã có trong file cache chính (đã gộp ở lần chạy trước) thì không chạy lại
        main_cache = out_dir / out_name
        if main_cache.exists():
            done = json.loads(main_cache.read_text(encoding="utf-8"))
            paths = [p for p in paths if image_key(p.parent, p.name) not in done]
        # mỗi tiến trình ghi file riêng (ghi chung một file sẽ đè nhau); gộp lại bằng merge_image_text_caches
        out_name = shard_cache_name(out_name, args.shard, args.num_shards)

    cache = build_image_text_cache(paths, out_dir / out_name, extractor, desc=args.task,
                                   batch_size=args.batch_size)
    keys = [image_key(p.parent, p.name) for p in paths]
    non_empty = sum(bool(cache.get(k)) for k in keys)
    missing = sum(k not in cache for k in keys)
    print(f"{args.task}: {non_empty}/{len(paths)} ảnh có text, lưu tại {out_dir / out_name}")
    if missing:
        sys.exit(f"{args.task}: {missing}/{len(paths)} ảnh lỗi, chưa có trong cache. Xem log rồi chạy lại để thử lại.")


if __name__ == "__main__":
    main()
