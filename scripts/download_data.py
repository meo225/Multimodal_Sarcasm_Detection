import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def get_auth():
    # 1. Check KAGGLE_API_TOKEN / access_token
    token = os.environ.get("KAGGLE_API_TOKEN")
    access_token_file = Path.home() / ".kaggle" / "access_token"
    if not token and access_token_file.exists():
        token = access_token_file.read_text(encoding="utf-8").strip()
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}, "auth": None}

    # 2. Check kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        kaggle_json = ROOT / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        return {"headers": {}, "auth": (data["username"], data["key"])}

    raise RuntimeError("Không tìm thấy Kaggle credentials (~/.kaggle/access_token hoặc ~/.kaggle/kaggle.json)")


def download_file_with_progress(url, dest_path, headers=None, auth=None):
    resp = requests.get(url, headers=headers, auth=auth, stream=True, allow_redirects=True, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Lỗi tải dataset (status code {resp.status_code}): {resp.text[:200]}")

    total_size = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, desc="Đang tải ViMMSD"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="slug dataset ViMMSD trên Kaggle, xem data/README.md")
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    auth_info = get_auth()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{args.dataset}"
    zip_path = out / "dataset_temp.zip"

    print(f"Bắt đầu tải dataset: {args.dataset}")
    download_file_with_progress(url, zip_path, headers=auth_info["headers"], auth=auth_info["auth"])

    print("\nĐang giải nén dữ liệu...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    zip_path.unlink()

    print(f"\nĐã tải và giải nén thành công vào {out}:")
    for p in sorted(out.iterdir()):
        if p.name != ".gitkeep":
            print("  ", p.name)


if __name__ == "__main__":
    main()
