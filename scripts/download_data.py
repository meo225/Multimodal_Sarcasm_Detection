"""Tải ViMMSD bằng Kaggle API (chỉ dùng khi chạy local; trên Kaggle thì attach dataset qua Add Input).

Cần file ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).
python scripts/download_data.py --dataset <owner>/<dataset-slug>"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="slug dataset ViMMSD trên Kaggle, xem data/README.md")
    parser.add_argument("--competition", action="store_true", help="slug là competition thay vì dataset")
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.competition:
        api.competition_download_files(args.dataset, path=out)
        for z in out.glob("*.zip"):
            import zipfile

            with zipfile.ZipFile(z) as f:
                f.extractall(out)
            z.unlink()
    else:
        api.dataset_download_files(args.dataset, path=out, unzip=True)
    print(f"đã tải về {out}:")
    for p in sorted(out.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
