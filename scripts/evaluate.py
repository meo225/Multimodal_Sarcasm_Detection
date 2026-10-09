"""Đánh giá checkpoint: python scripts/evaluate.py --config configs/fusion_concat.yaml [--split test]

--split test/val: xuất metrics, confusion matrix, file dự đoán (kèm xác suất) để phân tích lỗi.
--split public_test: tập không nhãn, xuất results.json theo định dạng nộp bài UIT DSC."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from vimmsd.data.dataset import ViMMSDCollator, build_dataloaders, build_datasets
from vimmsd.models.classifier import MODALITY_NEEDS, build_model
from vimmsd.training.metrics import compute_metrics, format_metrics, plot_confusion_matrix
from vimmsd.training.trainer import predict
from vimmsd.utils.checkpoint import load_checkpoint
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="mặc định: <output_dir>/<tên config>/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test", "public_test"])
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(args.config, args.override)
    output_dir = Path(cfg.paths.output_dir) / cfg.name
    ckpt = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    load_checkpoint(ckpt, model, map_location=device)
    model.to(device)

    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    datasets = build_datasets(cfg, needs_text, needs_image, splits=(args.split,))
    if args.split not in datasets:
        raise FileNotFoundError(f"không tìm thấy dữ liệu cho split {args.split}")
    collator = ViMMSDCollator(model.tokenizer, model.image_processor, cfg.data.max_length,
                              cfg.data.get("max_caption_length"))
    loader = build_dataloaders(cfg, datasets, collator)[args.split]
    out = predict(model, loader, device, use_amp=cfg.train.amp and device.type == "cuda")

    labels = list(cfg.data.labels)
    if args.split == "public_test":
        results = {"results": {i: labels[p] for i, p in zip(out["ids"], out["preds"])}, "phase": "dev"}
        path = output_dir / "results.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"đã ghi {len(out['ids'])} dự đoán vào {path}")
        return

    metrics = compute_metrics(out["labels"], out["preds"], labels)
    print(format_metrics(metrics))
    (output_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(metrics["confusion_matrix"], labels,
                          output_dir / f"confusion_matrix_{args.split}.png", title=f"{cfg.name} ({args.split})")

    texts = {r["id"]: r["caption"] for r in datasets[args.split].records}
    df = pd.DataFrame({"id": out["ids"], "label": [labels[i] for i in out["labels"]],
                       "pred": [labels[i] for i in out["preds"]]})
    df["caption"] = df["id"].map(texts)
    for j, name in enumerate(labels):
        df[f"prob_{name}"] = out["probs"][:, j].numpy()
    df.to_csv(output_dir / f"{args.split}_predictions.csv", index=False)
    print(f"đã ghi kết quả vào {output_dir}")


if __name__ == "__main__":
    main()
