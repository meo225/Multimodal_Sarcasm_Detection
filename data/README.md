# Dữ liệu ViMMSD

Không commit dữ liệu vào git. Bộ dữ liệu ViMMSD (UIT Data Science Challenge 2024) gồm:

| File / thư mục | Nội dung |
|---|---|
| `vimmsd-train.json` | annotation tập train: `{id: {"image", "caption", "label"}}` |
| `training-images/train-images/` | ảnh tập train |
| `vimmsd-public-test.json` | annotation public test (không có nhãn) |
| `public-test-images/dev-images/` | ảnh public test |

Tên file/thư mục trên phải khớp với mục `data` trong `configs/base.yaml`. Nếu bản tải về đặt tên khác thì sửa config, không đổi tên file.

Nguồn: Kaggle dataset [`hhhoang/vimmsd-dataset`](https://www.kaggle.com/datasets/hhhoang/vimmsd-dataset) (~1 GB). Có thêm `vimmsd-private-test.json` + `private-test-images/test-images/` (cũng không có nhãn).

Chạy `notebooks/00_dataset_check.ipynb` để kiểm tra dữ liệu trước khi train.

## Kaggle

Attach dataset qua **Add Input** trong notebook, sau đó sửa `paths.kaggle.data_dir` trong `configs/base.yaml` nếu đường dẫn khác `/kaggle/input/datasets/hhhoang/vimmsd-dataset` (đường dẫn đã xác nhận khi chạy notebook `00_dataset_check`).

## Local

```bash
# cần ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token)
python scripts/download_data.py --dataset hhhoang/vimmsd-dataset
```

Dữ liệu được tải về `data/raw/`.

## Colab

Upload dữ liệu lên Google Drive tại `MyDrive/IE403/data/raw/` (xem `paths.colab` trong `configs/base.yaml`).

## Chia tập

ViMMSD public test không có nhãn, nên train/val/test được chia **stratified** từ `vimmsd-train.json` (mặc định 80/10/10, seed trong config). Public test chỉ dùng để xuất file dự đoán: `python scripts/evaluate.py --config ... --split public_test`.
