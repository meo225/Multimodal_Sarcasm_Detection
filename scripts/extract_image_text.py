"""Tiền xử lý ảnh thành text, chạy offline một lần trước khi train (cần GPU cho VLM).

    python scripts/extract_image_text.py --config configs/base.yaml --task ocr
    python scripts/extract_image_text.py --config configs/base.yaml --task description --vlm vintern

Kết quả lưu vào <data.image_text.dir hoặc paths.cache_dir>/<ocr_cache | description_cache>.
Chạy tiếp được nếu bị ngắt: ảnh đã có trong file cache sẽ được bỏ qua."""
import argparse
import logging
from pathlib import Path

from vimmsd.data.image_text import OCRExtractor, VLMDescriber, build_image_text_cache, list_images
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--task", required=True, choices=["ocr", "description"])
    parser.add_argument("--vlm", default="vintern", choices=["vintern", "hf"], help="backend VLM cho --task description")
    parser.add_argument("--model", default=None, help="tên model HF, mặc định theo backend")
    parser.add_argument("--splits", nargs="*", default=["train", "public_test", "private_test"])
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu mỗi split (để thử)")
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

    it_cfg = cfg.data.image_text
    out_dir = Path(it_cfg.get("dir") or cfg.paths.cache_dir)
    if args.task == "ocr":
        extractor, out_name = OCRExtractor(lang="vi"), it_cfg.ocr_cache
    else:
        extractor, out_name = VLMDescriber(backend=args.vlm, model_name=args.model), it_cfg.description_cache

    cache = build_image_text_cache(paths, out_dir / out_name, extractor, desc=args.task)
    non_empty = sum(bool(cache.get(f"{p.parent.name}/{p.name}")) for p in paths)
    print(f"{args.task}: {non_empty}/{len(paths)} ảnh có text, lưu tại {out_dir / out_name}")


if __name__ == "__main__":
    main()
