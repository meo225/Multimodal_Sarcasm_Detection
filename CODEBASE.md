# CODEBASE.md — Thiết kế codebase cho dự án ViMMSD

Tài liệu này mô tả cấu trúc thư mục, module, quy ước code và cách môi trường chạy được thiết lập, để cả 4 thành viên code theo cùng một khung thay vì mỗi người viết một kiểu rồi khó ghép lại. Tham khảo `PLAN.md` để biết module nào cần cho tuần nào.

---

## 1. Nguyên tắc thiết kế

- **Tách logic khỏi notebook**: mọi hàm/class tái sử dụng (dataset, model, loss, metric...) viết trong `src/`, notebook chỉ dùng để gọi và chạy thử nghiệm/visualize. Lý do: 4 người cùng làm trên Colab/Kaggle, nếu code nằm hết trong notebook sẽ rất khó merge và dễ đụng nhau.
- **Config hóa thí nghiệm**: mỗi lần train (text-only, image-only, concat, cross-attention...) là 1 file config (YAML), không hard-code hyperparameter trong code. Giúp so sánh kết quả giữa các tuần dễ dàng và tái lặp được.
- **Không commit dữ liệu/checkpoint lớn lên git** — chỉ commit code + config. Data và model checkpoint lưu trên Google Drive / Kaggle Dataset riêng (xem mục 5).
- **Một entrypoint train duy nhất** (`scripts/train.py`), nhận vào 1 config, chạy được cả trên Colab lẫn Kaggle mà không sửa code.
- **Notebook tự bootstrap**: khi chạy trên Kaggle/Colab chỉ cần upload file `.ipynb`, không upload cả codebase. Cell đầu tiên của mỗi notebook tự `git clone` repo từ GitHub và `pip install -e` package `vimmsd` (xem mục 6). Code luôn lấy từ git nên cả nhóm chạy cùng một phiên bản, không có bản copy lệch nhau.
- **Không hard-code đường dẫn theo môi trường**: data/output path khai báo trong `configs/base.yaml` cho từng môi trường (local/Kaggle/Colab), code tự nhận diện môi trường đang chạy.

---

## 2. Cấu trúc thư mục

```
IE403/
├── README.md              # Mô tả dự án, bài toán, input/output
├── PLAN.md                 # Kế hoạch 8 tuần
├── CODEBASE.md             # (tài liệu này)
├── pyproject.toml          # khai báo package `vimmsd` để `pip install -e .`
├── requirements.txt        # Thư viện đầy đủ (chạy local)
├── requirements-kaggle.txt # Chỉ các thư viện Kaggle/Colab chưa có sẵn
├── .gitignore
│
├── configs/                 # 1 file config = 1 thí nghiệm
│   ├── base.yaml             # config gốc: seed, paths theo môi trường, batch size mặc định...
│   ├── text_only.yaml         # Tuần 3
│   ├── image_only.yaml        # Tuần 3
│   ├── fusion_concat.yaml     # Tuần 4
│   ├── fusion_cross_attn.yaml # Tuần 5
│   └── fusion_cross_attn_focal.yaml # Tuần 5 (+ xử lý imbalance)
│
├── data/
│   ├── README.md            # Hướng dẫn tải ViMMSD (script + link, không commit data thật)
│   ├── raw/                  # (gitignored) dữ liệu gốc tải về
│   └── processed/            # (gitignored) dữ liệu sau tiền xử lý (cache)
│
├── src/
│   └── vimmsd/
│       ├── __init__.py
│       ├── data/
│       │   ├── dataset.py         # class ViMMSDDataset (PyTorch Dataset)
│       │   ├── preprocessing.py   # word segmentation, chuẩn hóa text, resize/normalize ảnh
│       │   ├── image_text.py      # ảnh → text: OCR (PaddleOCR) + mô tả ảnh (VLM Vintern), cache JSON
│       │   └── augmentation.py    # back-translation, image augmentation nhẹ
│       │
│       ├── models/
│       │   ├── text_encoder.py    # wrapper PhoBERT
│       │   ├── image_encoder.py   # wrapper CLIP/ViT
│       │   ├── fusion.py          # ConcatFusion, CrossAttentionFusion
│       │   └── classifier.py      # head phân loại 4 lớp, lắp ghép encoder + fusion
│       │
│       ├── training/
│       │   ├── trainer.py         # vòng lặp train/eval chung
│       │   ├── losses.py          # CrossEntropy có weight, FocalLoss
│       │   └── metrics.py         # precision/recall/f1 (macro), confusion matrix
│       │
│       ├── analysis/              # Tuần 6-7
│       │   ├── shortcut_check.py    # thí nghiệm tráo ảnh
│       │   ├── interpretability.py  # Grad-CAM, attention map visualization
│       │   └── llm_compare.py       # gọi Vintern/LLM đa modal, so sánh kết quả
│       │
│       └── utils/
│           ├── config.py          # load/parse YAML config thành object, chọn paths theo môi trường
│           ├── env.py             # detect_env(): "kaggle" / "colab" / "local"
│           ├── seed.py            # set seed cho reproducibility
│           └── checkpoint.py      # save/load checkpoint (tương thích Drive/Kaggle)
│
├── scripts/
│   ├── download_data.py     # tải ViMMSD bằng Kaggle API (chỉ dùng khi chạy local)
│   ├── extract_image_text.py # tạo cache OCR / mô tả VLM cho toàn bộ ảnh (chạy 1 lần, cần GPU)
│   ├── train.py              # entrypoint train: `python train.py --config configs/xxx.yaml`
│   └── evaluate.py           # entrypoint eval trên test set + xuất confusion matrix
│
├── notebooks/                # Notebook cho từng giai đoạn (theo PLAN.md), upload thẳng lên Kaggle
│   ├── 00_template.ipynb      # notebook mẫu chứa sẵn cell bootstrap (mục 6), copy ra khi tạo notebook mới
│   ├── 00_dataset_check.ipynb # kiểm tra dữ liệu ViMMSD, chạy độc lập không cần clone repo
│   ├── 01_eda.ipynb
│   ├── 01b_image_text_extraction.ipynb  # chạy OCR + VLM, kiểm tra chất lượng text sinh ra
│   ├── 02_baseline_text.ipynb
│   ├── 03_baseline_image.ipynb
│   ├── 04_fusion_concat.ipynb
│   ├── 05_fusion_cross_attention.ipynb
│   ├── 06_shortcut_interpretability.ipynb
│   └── 07_ocr_llm_comparison.ipynb    # ablation OCR/mô tả VLM + so sánh LLM đa modal
│
├── reports/
│   ├── results.md            # bảng tổng hợp kết quả mọi thí nghiệm (cập nhật liên tục)
│   └── figures/               # hình ảnh xuất ra cho báo cáo (attention map, confusion matrix...)
│
└── tests/
    └── test_dataset.py       # test nhanh: load 1 batch, shape đúng, không lỗi encoding
```

---

## 3. Vai trò từng module (map với Tuần trong PLAN.md)

| Module | Dùng ở tuần | Mô tả |
|---|---|---|
| `data/preprocessing.py`, `data/dataset.py` | Tuần 1-2 | Load raw data, tách từ, chuẩn hóa, resize ảnh, trả về tensor sẵn sàng cho model |
| `data/image_text.py`, `scripts/extract_image_text.py` | Tuần 2 | OCR + mô tả ảnh bằng VLM, chạy 1 lần tạo cache `ocr.json`, `vlm_description.json`; bật bằng `data.image_text` trong config |
| `models/text_encoder.py`, `models/image_encoder.py` | Tuần 3 | Wrapper PhoBERT/CLIP dùng độc lập cho baseline |
| `models/fusion.py`, `models/classifier.py` | Tuần 4-5 | `ConcatFusion` (Tuần 4), `CrossAttentionFusion` (Tuần 5) — cùng interface để dễ swap trong config |
| `training/losses.py` | Tuần 5 | `FocalLoss`, weighted CrossEntropy cho lớp hiếm `text-sarcasm` |
| `training/trainer.py`, `training/metrics.py` | Tuần 3-8 | Vòng lặp train/eval dùng chung cho mọi thí nghiệm, tránh copy-paste code train ở từng notebook |
| `analysis/shortcut_check.py` | Tuần 6 | Swap ảnh giữa các mẫu, đo % thay đổi dự đoán |
| `analysis/interpretability.py` | Tuần 6 | Grad-CAM trên ảnh, attention weight trên text |
| `analysis/llm_compare.py` | Tuần 7 | Prompt Vintern/LLM đa modal, so sánh với model fine-tune |

---

## 4. Quy ước code

- **Python version**: 3.10+ (tương thích Colab/Kaggle mặc định).
- **Style**: theo PEP8, format bằng `black`, không bắt buộc nhưng khuyến khích chạy trước khi commit.
- **Naming**: file/module `snake_case`, class `PascalCase`, hằng số `UPPER_CASE`.
- **Docstring**: chỉ viết khi hàm có logic không hiển nhiên (ví dụ công thức focal loss, cách xử lý ảnh khi swap test) — không viết docstring cho hàm tự giải thích qua tên.
- **Commit message**: `<khu vực>: <mô tả ngắn>` — ví dụ `data: thêm xử lý teencode`, `model: cài cross-attention fusion`.
- **Notebook**: cell đầu tiên luôn là cell bootstrap copy từ `notebooks/00_template.ipynb`. Không định nghĩa class/hàm tái sử dụng trong notebook — nếu cần thì đưa vào `src/vimmsd/`, push lên GitHub, rồi notebook clone lại.
- **Branch**: mỗi người 1 nhánh theo phần việc (`feat/text-baseline`, `feat/cross-attention`...), merge vào `main` qua PR, tối thiểu 1 người khác review trước khi merge (tránh 1 người code lỗi làm hỏng pipeline chung).

---

## 5. Quản lý dữ liệu & checkpoint (do dùng GPU free)

- **Data**: không commit vào git.
  - **Kaggle**: attach bộ ViMMSD vào notebook qua *Add Input* (không cần tải bằng API), data nằm ở `/kaggle/input/<tên-dataset>/`.
  - **Colab**: để data trên Google Drive, mount Drive ở đầu notebook.
  - **Local**: dùng `scripts/download_data.py` (gọi Kaggle API) để tải về `data/raw/`.
- **Checkpoint model**: vì free tier hay bị ngắt session giữa chừng, `utils/checkpoint.py` cần hỗ trợ resume từ checkpoint gần nhất (qua key `resume_from` trong config).
  - **Kaggle**: lưu vào `/kaggle/working/outputs/`. Chỉ được giữ lại khi chạy bằng *Save Version → Save & Run All*. Muốn resume thì attach output của version trước làm input của notebook, rồi trỏ `resume_from` vào đó.
  - **Colab**: lưu thẳng lên Google Drive.
- **Kết quả thí nghiệm**: mỗi lần train xong, ghi 1 dòng vào `reports/results.md` (config dùng, **commit hash** của code, macro-F1, thời gian train) để cả nhóm theo dõi tiến độ chung mà không cần hỏi nhau qua chat. Commit hash giúp tái lặp đúng kết quả, vì notebook luôn clone code mới nhất.

---

## 6. Chạy trên Kaggle/Colab chỉ bằng notebook

Mục tiêu: chỉ upload file `.ipynb` lên Kaggle, code còn lại notebook tự lấy từ GitHub.

### 6.1. Chuẩn bị một lần (mỗi thành viên)

1. Tài khoản Kaggle đã xác minh số điện thoại (bắt buộc để bật *Internet* trong notebook).
2. Nếu repo GitHub là **private**: tạo GitHub fine-grained token (quyền read-only, chỉ cho repo này), lưu vào **Kaggle → Add-ons → Secrets** với tên `GITHUB_TOKEN`. Tuyệt đối không dán token trực tiếp vào notebook.
3. Trong notebook: bật *Internet: On*, *Accelerator: GPU*, và attach dataset ViMMSD.

### 6.2. Cell bootstrap (cell đầu tiên của mọi notebook)

Bản gốc nằm trong `notebooks/00_template.ipynb` (cell 2). Tạo notebook mới thì copy từ đó:

```python
# === Cell bootstrap (giống nhau ở mọi notebook, xem CODEBASE.md mục 6) ===
import os, subprocess, sys
from pathlib import Path

REPO = "github.com/yunaLee21/Multimodal_Sarcasm_Detection.git"
REF = "main"               # branch của mình, hoặc 1 commit hash để tái lặp kết quả
CODE_DIR = "/tmp/IE403"    # clone ra ngoài /kaggle/working để token không bị lưu vào output


def get_github_token():
    """Token cho repo private; repo public thì không cần tạo secret."""
    try:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("GITHUB_TOKEN")
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


ON_CLOUD = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or "COLAB_RELEASE_TAG" in os.environ
if ON_CLOUD:
    if "COLAB_RELEASE_TAG" in os.environ:
        from google.colab import drive
        drive.mount("/content/drive")
    token = get_github_token()
    url = f"https://{token}@{REPO}" if token else f"https://{REPO}"
    subprocess.run(["rm", "-rf", CODE_DIR])
    subprocess.run(["git", "clone", "-q", url, CODE_DIR], check=True)
    subprocess.run(["git", "-C", CODE_DIR, "checkout", "-q", REF], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE_DIR}/requirements-kaggle.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", CODE_DIR], check=True)
    if CODE_DIR + "/src" not in sys.path:
        sys.path.insert(0, CODE_DIR + "/src")  # kernel đang chạy chưa thấy package vừa cài -e
    os.chdir(CODE_DIR)
elif Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)  # local: chạy từ gốc repo (đã `pip install -e .` một lần)

print("cwd:", os.getcwd())
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code commit:", commit or "unknown")
```

Repo public thì không cần tạo secret `GITHUB_TOKEN`, cell sẽ tự clone không kèm token. Trên Colab, cell tự mount Google Drive.

Sau cell này, notebook dùng code như bình thường:

```python
from vimmsd.utils.config import load_config
cfg = load_config("configs/text_only.yaml")   # tự chọn paths theo môi trường

# hoặc chạy entrypoint train:
!python scripts/train.py --config configs/text_only.yaml
```

### 6.3. Đường dẫn theo môi trường

`configs/base.yaml` khai báo paths cho từng môi trường, `utils/env.py` nhận diện môi trường (`KAGGLE_KERNEL_RUN_TYPE` → kaggle, `COLAB_RELEASE_TAG` → colab, còn lại → local) và `utils/config.py` chọn đúng nhánh:

```yaml
paths:
  local:
    data_dir: data/raw          # đường dẫn tương đối tính từ gốc repo
    output_dir: outputs
    cache_dir: data/processed
  kaggle:
    data_dir: /kaggle/input/<tên-dataset-vimmsd>
    output_dir: /kaggle/working/outputs
    cache_dir: /kaggle/working/cache
  colab:
    data_dir: /content/drive/MyDrive/IE403/data/raw
    output_dir: /content/drive/MyDrive/IE403/outputs
    cache_dir: /content/drive/MyDrive/IE403/cache
```

Mọi giá trị trong config đều ghi đè được từ dòng lệnh, kể cả tên thí nghiệm (quyết định thư mục output):

```bash
python scripts/train.py --config configs/fusion_cross_attn.yaml \
    --override name=xattn_oversample data.sampler=weighted train.epochs=5
```

### 6.4. Quy trình làm việc hằng ngày

1. Sửa code trong `src/` trên máy local → commit → push lên branch của mình.
2. Trên Kaggle, đặt `REF` = tên branch đó, *Run All*. Notebook tự clone code mới nhất, không cần upload lại gì.
3. Train chính thức thì đặt `REF` = commit hash cụ thể và ghi hash vào `reports/results.md`.

---

## 7. Môi trường & dependencies

`pyproject.toml` (tối thiểu, để `pip install -e .` cài được package `vimmsd` từ `src/`):

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vimmsd"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

`requirements.txt` (danh sách đầy đủ, dùng khi chạy local):

```
torch
torchvision
transformers>=4.45
underthesea          # tách từ tiếng Việt cho PhoBERT
emoji                # xử lý emoji trong caption
scikit-learn         # metrics
pyyaml               # đọc config
pandas
matplotlib
pillow
tqdm
paddleocr            # OCR ở bước tiền xử lý ảnh
paddlepaddle         # backend cho paddleocr
kaggle               # download data qua API (chỉ dùng local)
pytest
```

CLIP được load qua `transformers` (`CLIPVisionModel`, checkpoint `openai/clip-vit-base-patch32`), không cần package `clip` của OpenAI. Nhờ vậy lấy được embedding từng patch cho cross-attention, và đổi sang ViT khác chỉ cần sửa tên model trong config.

`requirements-kaggle.txt` (dùng trong cell bootstrap). Kaggle/Colab đã cài sẵn `torch`, `torchvision`, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`. **Không cài lại `torch`** vì dễ làm lệch phiên bản CUDA của môi trường:

```
underthesea
emoji
```

`paddleocr` nặng và chỉ cần khi tạo cache ảnh → text (chạy 1 lần) nên không nằm trong file này. `notebooks/01b_image_text_extraction.ipynb` tự cài `paddlepaddle-gpu paddleocr timm einops` (`timm`, `einops` cần cho Vintern). Các notebook train chỉ đọc cache JSON nên không cần cài OCR/VLM.

Không cần môi trường ảo phức tạp. Kaggle/Colab cài qua cell bootstrap, còn local thì chạy `pip install -r requirements.txt && pip install -e .` một lần. Kiểm tra nhanh (không cần GPU/mạng): `pytest tests`.

---

## 8. Bước tiếp theo

Khung code ở mục 2 đã có đủ (trừ `PLAN.md`). Việc cần làm để chạy thật:

1. ~~`git init`, push lên GitHub, sửa `REPO` trong cell bootstrap~~ (đã xong: repo public [yunaLee21/Multimodal_Sarcasm_Detection](https://github.com/yunaLee21/Multimodal_Sarcasm_Detection), không cần `GITHUB_TOKEN`).
2. Chạy `notebooks/00_dataset_check.ipynb` trên Kaggle (attach `hhhoang/vimmsd-dataset`), đối chiếu đường dẫn nó in ra với `paths.kaggle.data_dir` trong `configs/base.yaml`.
3. Xử lý các điểm "CHÚ Ý" trong bảng tổng kết của notebook (tiền xử lý caption, cách chia val/test).
4. Chạy `notebooks/00_template.ipynb` trên Kaggle để chắc chắn clone + import `vimmsd` + đọc dữ liệu hoạt động, rồi chạy `01_eda.ipynb` và `01b_image_text_extraction.ipynb` (tạo cache OCR + mô tả VLM, cần GPU).
5. Sau đó các thành viên làm song song theo notebook 02–07, chỉnh hyperparameter qua config/`--override`, sửa code trong `src/` qua PR.
