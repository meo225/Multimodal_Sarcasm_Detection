"""Entrypoint train: python scripts/train.py --config configs/text_only.yaml [--override train.lr_head=5e-4 ...]"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from vimmsd.data.dataset import ViMMSDCollator, build_dataloaders, build_datasets, class_counts
from vimmsd.models.classifier import MODALITY_NEEDS, build_model
from vimmsd.training.losses import build_loss
from vimmsd.training.metrics import format_metrics, plot_confusion_matrix
from vimmsd.training.trainer import Trainer
from vimmsd.utils.config import find_project_root, load_config
from vimmsd.utils.env import git_commit
from vimmsd.utils.seed import set_seed

RESULTS_HEADER = (
    "| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian train (phút) | Ghi chú |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def append_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        path.write_text("# Kết quả thí nghiệm\n\n" + RESULTS_HEADER, encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[], help='ví dụ: train.epochs=3 loss.name=focal')
    parser.add_argument("--note", default="", help="ghi chú thêm vào reports/results.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("train")

    cfg = load_config(args.config, args.override)
    set_seed(cfg.seed)
    commit = git_commit()
    output_dir = Path(cfg.paths.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "commit": commit}, indent=2))
    log.info("config %s | env %s | commit %s | output %s", cfg.config_path, cfg.env, commit, output_dir)

    model = build_model(cfg)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    datasets = build_datasets(cfg, needs_text, needs_image)
    collator = ViMMSDCollator(model.tokenizer, model.image_processor, cfg.data.max_length,
                              cfg.data.get("max_caption_length"))
    loaders = build_dataloaders(cfg, datasets, collator)
    counts = class_counts(datasets["train"], len(cfg.data.labels))
    log.info("train %d | val %d | test %d | phân bố lớp train %s",
             len(datasets["train"]), len(datasets["val"]), len(datasets["test"]),
             dict(zip(cfg.data.labels, counts)))

    trainer = Trainer(model, build_loss(cfg.loss, counts), loaders, cfg, output_dir)
    summary = trainer.fit()

    trainer.load_best()
    test_metrics, _ = trainer.evaluate("test")
    log.info("test (best checkpoint)\n%s", format_metrics(test_metrics))
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    plot_confusion_matrix(test_metrics["confusion_matrix"], cfg.data.labels,
                          output_dir / "confusion_matrix_test.png", title=cfg.name)

    row = (f"| {datetime.now():%Y-%m-%d} | `{cfg.name}` | `{commit}` | {cfg.env} | "
           f"{summary['best_val_macro_f1']:.4f} | {test_metrics['macro_f1']:.4f} | "
           f"{summary['train_minutes']:.1f} | {args.note} |")
    # trên Kaggle/Colab repo là bản clone tạm, nên ghi thêm vào output_dir để copy dòng này về reports/results.md
    append_result(output_dir / "results_row.md", row)
    if cfg.env == "local":
        append_result(find_project_root(args.config) / "reports" / "results.md", row)
    print("\nDòng kết quả cho reports/results.md:\n" + row)


if __name__ == "__main__":
    main()
