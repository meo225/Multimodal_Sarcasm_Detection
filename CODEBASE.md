# CODEBASE.md — Thiết kế codebase cho dự án ViMMSD

Tài liệu này mô tả cấu trúc thư mục, module, quy ước code và cách môi trường chạy được thiết lập, để cả 4 thành viên code theo cùng một khung thay vì mỗi người viết một kiểu rồi khó ghép lại. Tham khảo `PLAN.md` để biết module nào cần cho tuần nào.

---

## 1. Nguyên tắc thiết kế

- **Tách logic khỏi notebook**: mọi hàm/class tái sử dụng (dataset, model, loss, metric...) viết trong `src/`, notebook chỉ dùng để gọi và chạy thử nghiệm/visualize. Lý do: 4 người cùng làm trên Colab/Kaggle, nếu code nằm hết trong notebook sẽ rất khó merge và dễ đụng nhau.
- **Config hóa thí nghiệm**: mỗi lần train (text-only, image-only, concat, cross-attention...) là 1 file config (YAML), không hard-code hyperparameter trong code. Giúp so sánh kết quả giữa các tuần dễ dàng và tái lặp được.
- **Không commit dữ liệu/checkpoint lớn lên git** — chỉ commit code + config. Data và model checkpoint lưu trên Google Drive / Kaggle Dataset riêng (xem mục 5).
- **Một entrypoint train duy nhất** (`scripts/train.py`), nhận vào 1 config, chạy được trên Kaggle, Colab và GPU server mà không sửa code.
- **Mỗi bước đều có script `.py`**: tiền xử lý text, OCR / mô tả ảnh, sinh reasoning, train, đánh giá. Notebook chỉ gọi lại các script hoặc hàm trong `src/`. Nhờ vậy trên GPU server chỉ cần chạy `.py`, không cần notebook (mục 7).
- **Notebook tự bootstrap**: khi chạy trên Kaggle/Colab chỉ cần upload file `.ipynb`, không upload cả codebase. Cell đầu tiên của mỗi notebook tự `git clone` repo từ GitHub và `pip install -e` package `vimmsd` (xem mục 6). Code luôn lấy từ git nên cả nhóm chạy cùng một phiên bản, không có bản copy lệch nhau.
- **Không hard-code đường dẫn theo môi trường**: data/output path khai báo trong `configs/base.yaml` cho từng môi trường (local/server/Kaggle/Colab), code tự nhận diện môi trường đang chạy.

---

## 2. Cấu trúc thư mục

```
IE403/
├── README.md              # Mô tả dự án, bài toán, input/output
├── PLAN.md                 # Kế hoạch 8 tuần
├── CODEBASE.md             # (tài liệu này)
├── docs/
│   ├── cot_distillation.md       # thiết kế M1 (CoT distillation) và M2 (+ hierarchical)
│   └── cot_distillation_plan.md  # kế hoạch chạy M1/M2 trên GPU server
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
│   ├── fusion_concat_frozen.yaml # Tuần 4: đóng băng cả 2 encoder, chỉ train MLP head
│   ├── fusion_concat_ftlast.yaml # Tuần 4: như trên + fine-tune nhẹ 2 block cuối mỗi encoder
│   ├── text_only_frozen.yaml  # Tuần 4: mốc so sánh, PhoBERT đóng băng
│   ├── image_only_frozen.yaml # Tuần 4: mốc so sánh, CLIP đóng băng
│   ├── text_only_frozen_cw.yaml # Tuần 4: text_only_frozen + class weight sqrt_inverse
│   ├── fusion_concat_frozen_reg.yaml # Tuần 4: concat đóng băng + LayerNorm/projection mỗi nhánh, dropout ảnh, modality dropout
│   ├── fusion_concat_frozen_reg_cw.yaml # Tuần 4: như trên + class weight sqrt_inverse
│   ├── fusion_multistream_frozen.yaml # Tuần 4: caption, OCR, mô tả VLM mã hoá riêng bằng PhoBERT đóng băng, nối vector
│   ├── fusion_multistream_clip_frozen.yaml # Tuần 4: như trên + vector CLIP
│   ├── fusion_multistream_ocr_frozen.yaml # Tuần 4: ablation, caption + OCR
│   ├── fusion_multistream_desc_frozen.yaml # Tuần 4: ablation, caption + mô tả VLM
│   ├── fusion_cross_attn.yaml # Tuần 5 (direction: both)
│   ├── fusion_cross_attn_t2i.yaml # Tuần 5: text attend vào ảnh
│   ├── fusion_cross_attn_i2t.yaml # Tuần 5: ảnh attend vào text
│   ├── fusion_cross_attn_focal.yaml # Tuần 5 (+ xử lý imbalance)
│   ├── distill_base.yaml           # backbone chung cho CoT distillation (multistream_clip đóng băng)
│   ├── distill_flat_{a,b,c,d}.yaml # M1 (= d) và các cột ablation, xem docs/cot_distillation.md mục 6.2; *_d_nogroup3 bỏ nhóm 3
│   ├── distill_hier_{a,b,c,d}.yaml # M2 (= d) và các cột ablation; *_d_nogroup3 bỏ nhóm 3
│   ├── hier_cascade.yaml           # kiểu pipeline phân cấp nối tiếp của Vu et al. (định tuyến cứng), để đối chiếu
│   └── experiments/                # plan config × seed cho run_experiments.py (mục 7.3)
│       ├── smoke.yaml              # kiểm tra server: 1 run, 1 epoch
│       ├── phase4_baselines.yaml   # 11 config notebook 04 × 3 seed, mốc cho cổng G0
│       ├── cot_distill_stage0.yaml # cột (a)/(b) + cascade × 3 seed, cổng G0
│       ├── cot_distill_grid.yaml   # grid λ_axis × λ_rat trên cột (d)
│       └── cot_distill.yaml        # ma trận đầy đủ × 3 seed
│
├── data/
│   ├── README.md            # Hướng dẫn tải ViMMSD (script + link, không commit data thật)
│   ├── raw/                  # (gitignored) dữ liệu gốc tải về
│   ├── processed/            # (gitignored) cache tiền xử lý: text_preprocessed.json, ocr.json,
│   │                         #   description.json, image_preprocessed.json
│   └── server/               # (gitignored) GPU server: symlink raw/, cache/, outputs/ ra ổ dữ liệu (mục 7.2)
│
├── src/
│   └── vimmsd/
│       ├── __init__.py
│       ├── data/
│       │   ├── dataset.py         # class ViMMSDDataset (PyTorch Dataset)
│       │   ├── preprocessing.py   # word segmentation, chuẩn hóa text, resize/normalize ảnh
│       │   ├── image_text.py      # ảnh → text: OCR (PaddleOCR + VietOCR) + mô tả ảnh (VLM Vintern), cache JSON
│       │   ├── rationale.py       # prompt CoT, teacher Qwen2.5-VL, parse JSON, phân nhóm, chấm (κ), gộp rationale_dataset.json
│       │   └── augmentation.py    # back-translation, image augmentation nhẹ
│       │
│       ├── models/
│       │   ├── text_encoder.py    # wrapper PhoBERT
│       │   ├── image_encoder.py   # wrapper CLIP/ViT
│       │   ├── freezing.py        # đóng băng encoder, mở lại N block cuối (unfreeze_last_n)
│       │   ├── fusion.py          # ConcatFusion, CrossAttentionFusion, MultiStreamFusion
│       │   ├── heads.py           # head flat / hierarchical (S × C) / cascade, hiệu chỉnh prior theo tầng
│       │   └── classifier.py      # lắp ghép encoder + fusion + head (model.head), head phụ T/I, phép chiếu reasoning
│       │
│       ├── training/
│       │   ├── trainer.py         # vòng lặp train/eval chung
│       │   ├── losses.py          # CE có weight, FocalLoss, SarcasmLoss (λ_S, λ_axis, λ_rat; loss cascade)
│       │   └── metrics.py         # precision/recall/f1 (macro), confusion matrix
│       │
│       ├── analysis/              # Tuần 6-7
│       │   ├── shortcut_check.py    # thí nghiệm tráo ảnh
│       │   ├── interpretability.py  # Grad-CAM, attention map visualization
│       │   └── llm_compare.py       # gọi Vintern/LLM đa modal, so sánh kết quả
│       │
│       └── utils/
│           ├── config.py          # load/parse YAML config thành object, chọn paths theo môi trường
│           ├── env.py             # detect_env(): VIMMSD_ENV nếu có ("server"), không thì "kaggle" / "colab" / "local"
│           ├── experiments.py     # plan config × seed: dựng job cho gpu_queue, tổng hợp kết quả các seed
│           ├── seed.py            # set seed cho reproducibility
│           ├── checkpoint.py      # save/load checkpoint (tương thích Drive/Kaggle)
│           └── gpu_queue.py       # chạy nhiều run song song, mỗi GPU một run (Kaggle T4 x2 hoặc GPU server)
│
├── scripts/                  # mỗi bước một script; trên GPU server chỉ dùng thư mục này (mục 7)
│   ├── download_data.py      # tải ViMMSD (và bộ cache đã preprocess) bằng Kaggle API
│   ├── preprocess_text.py    # chuẩn hoá toàn bộ caption → text_preprocessed.json (thay notebook 01a)
│   ├── extract_image_text.py # tạo cache OCR / mô tả VLM cho toàn bộ ảnh (chạy 1 lần, cần GPU)
│   ├── build_image_text_dataset.py # gộp ocr.json + description.json → image_preprocessed.json/.csv
│   ├── make_id_lists.py      # danh sách id cho tập thử teacher và các lô sinh reasoning
│   ├── generate_rationales.py      # teacher sinh k reasoning/mẫu, chạy tiếp được, chia shard
│   ├── eval_teacher.py       # so output teacher với nhãn: confusion matrix, F1, tỉ lệ parse lỗi
│   ├── build_rationale_dataset.py  # group / rationalize / export CSV chấm / merge → rationale_dataset.json
│   ├── embed_rationales.py   # mã hoá reasoning bằng PhoBERT đóng băng → rationale_embeddings.npy
│   ├── train.py              # entrypoint train: `python train.py --config configs/xxx.yaml`
│   ├── run_experiments.py    # chạy plan config × seed trên N GPU (dùng gpu_queue), ghi log + status
│   ├── summarize_results.py  # gộp metrics các seed → bảng trung bình ± độ lệch chuẩn (mặc định chỉ val)
│   ├── evaluate.py           # entrypoint eval trên test set + xuất confusion matrix
│   └── server/               # shell script cho GPU server
│       ├── setup_env.sh        # tạo venv, cài requirements, kiểm tra GPU / Java, chạy pytest (mục 7.1)
│       └── prepare_data.sh     # tải data + cache, tạo symlink data/server/*, hoặc sinh lại cache (mục 7.4)
│
├── notebooks/                # Notebook cho từng giai đoạn (theo PLAN.md), upload thẳng lên Kaggle; GPU server không dùng
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
│   ├── results.md            # bảng tổng hợp kết quả mọi thí nghiệm (cập nhật liên tục, cột Môi trường ghi kaggle/server)
│   ├── literature_review/    # review tài liệu; papers/ (PDF) gitignored
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
| `data/image_text.py`, `scripts/extract_image_text.py` | Tuần 2 | OCR + mô tả ảnh bằng VLM, chạy 1 lần tạo cache `ocr.json`, `description.json`, rồi `scripts/build_image_text_dataset.py` gộp thành `image_preprocessed.json` (kèm model đã dùng trong `meta`) và `image_preprocessed.csv`; bật bằng `data.image_text` trong config |
| `models/text_encoder.py`, `models/image_encoder.py` | Tuần 3 | Wrapper PhoBERT/CLIP dùng độc lập cho baseline |
| `models/fusion.py`, `models/classifier.py` | Tuần 4-5 | `ConcatFusion` (Tuần 4), `CrossAttentionFusion` (Tuần 5) — cùng interface để dễ swap trong config |
| `training/losses.py` | Tuần 5 | `FocalLoss`, weighted CrossEntropy cho lớp hiếm `text-sarcasm` |
| `training/trainer.py`, `training/metrics.py` | Tuần 3-8 | Vòng lặp train/eval dùng chung cho mọi thí nghiệm, tránh copy-paste code train ở từng notebook |
| `analysis/shortcut_check.py` | Tuần 6 | Swap ảnh giữa các mẫu, đo % thay đổi dự đoán |
| `analysis/interpretability.py` | Tuần 6 | Grad-CAM trên ảnh, attention weight trên text |
| `analysis/llm_compare.py` | Tuần 7 | Prompt Vintern/LLM đa modal, so sánh với model fine-tune |
| `data/rationale.py`, `scripts/generate_rationales.py`, `scripts/build_rationale_dataset.py`, `scripts/embed_rationales.py` | CoT distillation | Sinh, lọc, đóng gói reasoning của teacher. Chỉ chạy trên GPU server (`docs/cot_distillation_plan.md`) |
| `scripts/run_experiments.py`, `scripts/summarize_results.py` | CoT distillation | Thay cho cell chạy nhiều run trong notebook 04: chạy ma trận config × seed trên N GPU và tổng hợp kết quả |

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
  - **GPU server**: `scripts/server/prepare_data.sh` tải về ổ dữ liệu và symlink vào `data/server/` (mục 7.2).
  - **Kaggle**: attach bộ ViMMSD vào notebook qua *Add Input* (không cần tải bằng API), data nằm ở `/kaggle/input/<tên-dataset>/`.
  - **Colab**: để data trên Google Drive, mount Drive ở đầu notebook.
  - **Local**: dùng `scripts/download_data.py` (gọi Kaggle API) để tải về `data/raw/`.
- **Checkpoint model**: vì free tier hay bị ngắt session giữa chừng, `utils/checkpoint.py` cần hỗ trợ resume từ checkpoint gần nhất (qua key `resume_from` trong config).
  - **Kaggle**: lưu vào `/kaggle/working/outputs/`. Chỉ được giữ lại khi chạy bằng *Save Version → Save & Run All*. Muốn resume thì attach output của version trước làm input của notebook, rồi trỏ `resume_from` vào đó.
  - **Colab**: lưu thẳng lên Google Drive.
  - **GPU server**: lưu vào `data/server/outputs/<tên thí nghiệm>/` (symlink ra ổ dữ liệu). Không bị ngắt session nên chỉ cần `resume_from` khi tự dừng.
- **Kết quả thí nghiệm**: mỗi lần train xong, ghi 1 dòng vào `reports/results.md` (config dùng, **commit hash** của code, môi trường, macro-F1, thời gian train) để cả nhóm theo dõi tiến độ chung mà không cần hỏi nhau qua chat. Commit hash giúp tái lặp đúng kết quả, vì notebook luôn clone code mới nhất.
- **Chia sẻ cache giữa server và Kaggle**: cache sinh trên server (`text_preprocessed.json`, `image_preprocessed.json`, `rationale_dataset.json`) được upload thành phiên bản mới của Kaggle dataset, để các thành viên chạy Kaggle dùng đúng cùng dữ liệu (mục 7.6).

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
    if "COLAB_RELEASE_TAG" in os.environ and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):  # ảnh Kaggle dựng từ ảnh Colab nên cũng có biến này
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

`configs/base.yaml` khai báo paths cho từng môi trường, `utils/env.py` nhận diện môi trường (`KAGGLE_KERNEL_RUN_TYPE` → kaggle, `COLAB_RELEASE_TAG` → colab, còn lại → local; biến `VIMMSD_ENV` nếu có thì được ưu tiên, vd `server` trên GPU server, xem mục 7.2) và `utils/config.py` chọn đúng nhánh:

```yaml
paths:
  local:
    data_dir: data/raw          # đường dẫn tương đối tính từ gốc repo
    output_dir: outputs
    cache_dir: data/processed
  kaggle:
    data_dir: /kaggle/input/datasets/phuclkz/vimmsd-private-dataset
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

## 7. Chạy trên GPU server (chỉ `.py`)

Dùng cho người có quyền vào GPU server riêng; hiện là task CoT distillation (`docs/cot_distillation_plan.md`). Không dùng notebook: mọi bước đều là script trong `scripts/`, chạy trong `tmux` để không mất tiến trình khi ngắt SSH. Các thành viên khác vẫn chạy Kaggle như mục 6. Hai bên dùng chung code (git) và chung cache (Kaggle dataset).

### 7.1. Cài đặt một lần

```bash
git clone https://github.com/yunaLee21/Multimodal_Sarcasm_Detection.git ~/IE403
cd ~/IE403 && git checkout thanhtruc-main
TORCH_INDEX_URL=https://download.pytorch.org/whl/<cuXXX> bash scripts/server/setup_env.sh   # <cuXXX> khớp nvidia-smi
source .venv/bin/activate
```

`setup_env.sh` tạo `.venv`, cài `torch` (theo `TORCH_INDEX_URL` nếu có), `requirements.txt` trừ paddle bản CPU, `pip install -e .`, rồi in danh sách GPU, kiểm tra Java và chạy `pytest tests -q`. Thêm `WITH_IMAGE_TEXT=1` để cài cả OCR (`paddlepaddle-gpu`, `paddleocr`, `vietocr`) và `timm` cho Vintern. Chỉ cần khi sinh lại cache ảnh → text, vì train chỉ đọc JSON.

Biến môi trường đặt trong `~/.bashrc` (hoặc `.env` trong repo, không commit):

```bash
export VIMMSD_ENV=server                   # chọn nhánh paths.server trong configs/base.yaml
export HF_HOME=/data/<user>/hf_cache       # cache model HuggingFace, nên đặt ở ổ lớn
export VNCORENLP_DIR=/data/<user>/vncorenlp
```

Tải dataset Kaggle private cần `~/.kaggle/kaggle.json` hoặc `~/.kaggle/access_token` (`scripts/download_data.py` đọc từ đó). Thay `/data/<user>` bằng ổ dữ liệu thật của server.

### 7.2. Bố cục dữ liệu trên server

Dữ liệu, cache và checkpoint nằm trên ổ dữ liệu, repo trỏ tới bằng 3 symlink trong `data/server/` (gitignored). `paths.server` trong `configs/base.yaml` dùng đúng 3 đường dẫn này:

```yaml
paths:
  server:
    data_dir: data/server/raw
    output_dir: data/server/outputs
    cache_dir: data/server/cache
```

```
/data/<user>/                    # DATA_ROOT
├── vimmsd/                      # ViMMSD gốc (phuclkz/vimmsd-private-dataset)
│   ├── vimmsd-train.json, vimmsd-{public,private}-test{,-ref}.json, vimmsd-warmup.json
│   └── training-images/ public-test-images/ private-test-images/ warmup-images/
├── vimmsd_cache/
│   ├── text_preprocessed.json
│   ├── ocr.json, description.json, image_preprocessed.json/.csv
│   └── rationales/              # CoT distillation (docs/cot_distillation_plan.md)
│       ├── ids/*.txt                         # danh sách id tập thử và các lô
│       ├── raw/<prompt_version>/*.jsonl      # output teacher, chia shard
│       ├── review/<batch>.csv                # file chấm rubric
│       ├── rationale_dataset.json
│       └── rationale_embeddings.npy (+ ids)
├── vimmsd_outputs/              # checkpoint, metrics, run.log của từng run; _plans/<plan>/ là bảng tổng hợp
├── hf_cache/                    # HF_HOME
└── vncorenlp/                   # VNCORENLP_DIR

~/IE403/                         # repo clone
├── data/server/raw -> /data/<user>/vimmsd/...   (thư mục chứa vimmsd-train.json)
├── data/server/cache -> /data/<user>/vimmsd_cache
├── data/server/outputs -> /data/<user>/vimmsd_outputs
└── logs/*.log                   # log tiền xử lý của prepare_data.sh (gitignored)
```

`scripts/server/prepare_data.sh` tạo các thư mục và symlink này (mục 7.4). `data/raw/` và `data/processed/` của máy local không bị đụng tới.

### 7.3. Chạy tiến trình dài

```bash
tmux new -s vimmsd                          # mở lại: tmux attach -t vimmsd
```

- **Sinh dữ liệu** (`extract_image_text.py`, sau này là `generate_rationales.py`): mỗi GPU một tiến trình, chia việc bằng `--num-shards N --shard i`, mỗi shard một file cache, sau đó gộp lại. Bị dừng giữa chừng thì chạy lại đúng lệnh đó, mẫu đã có trong cache sẽ được bỏ qua.
- **Train nhiều run**: `scripts/run_experiments.py --plan configs/experiments/<tên>.yaml` chạy ma trận config × seed. Định dạng plan xem docstring của `src/vimmsd/utils/experiments.py`. Mỗi GPU một run, dùng `utils/gpu_queue.run_on_gpus`, thay cho cell chạy song song của notebook 04.
  - `--gpu-ids 0,2`: chỉ dùng các GPU này (server dùng chung). `--only <chuỗi>`: chỉ chạy run có tên chứa chuỗi đó. `--dry-run`: in lệnh, không chạy.
  - Run đã có `test_metrics.json` và các file `<split>_predictions.csv` thì bỏ qua (`--rerun` để chạy lại). Run xong thì xoá `last.pt`.
  - Log mỗi run ở `data/server/outputs/<run>/run.log`. Trạng thái cả plan ở `data/server/outputs/_plans/<plan>/status.json`.
  - Chạy xong tự tổng hợp **trên val**: `runs.csv`, `summary.csv`, `summary.md` (trung bình ± độ lệch chuẩn theo seed, gồm macro-F1, weighted-F1 và macro-F1 bỏ `text-sarcasm`).
- **Tổng hợp lại**: `scripts/summarize_results.py --plan ...`. Thêm `--with-test` để có cột private test, chỉ dùng ở bước đánh giá cuối.

Plan có sẵn:

| Plan | Nội dung |
|---|---|
| `configs/experiments/smoke.yaml` | 1 run `text_only_frozen` 1 epoch, kiểm tra server chạy được train + evaluate |
| `configs/experiments/phase4_baselines.yaml` | 11 config của notebook 04 × 3 seed, làm mốc cho cổng G0 |

### 7.4. Tiền xử lý trên server

Thay cho notebook 01a/01b. Chạy sau `setup_env.sh`, trong venv:

```bash
DATA_ROOT=/data/<user> bash scripts/server/prepare_data.sh               # tải ViMMSD + cache có sẵn của nhóm
DATA_ROOT=/data/<user> REGENERATE=1 bash scripts/server/prepare_data.sh  # sinh lại cache text + ảnh
```

| Bước | Lệnh trong `prepare_data.sh` | Đầu ra | GPU |
|---|---|---|---|
| 1. Tải dữ liệu, tạo symlink | `download_data.py --dataset phuclkz/vimmsd-private-dataset` | ViMMSD gốc, `data/server/*` | không |
| 2a. Tải cache có sẵn (mặc định) | `download_data.py --dataset funalee/vimmsd-data-preprocessed` | `text_preprocessed.json`, `image_preprocessed.json` | không |
| 2b. Tiền xử lý caption (`REGENERATE=1`) | `preprocess_text.py` (gọi `dataset.build_text_cache`) | `text_preprocessed.json` | không (VnCoreNLP chạy CPU) |
| 3. OCR (`REGENERATE=1`) | `extract_image_text.py --task ocr`, mỗi GPU một shard, rồi gộp | `ocr.json` | có |
| 4. Mô tả ảnh (`REGENERATE=1`) | `extract_image_text.py --task description --vlm $VLM`, mỗi GPU một shard, rồi gộp | `description.json` | có |
| 5. Gộp (`REGENERATE=1`) | `build_image_text_dataset.py --out-dir data/server/cache` | `image_preprocessed.json/.csv` | không |

Script chạy lại được: bước nào đã có kết quả thì bỏ qua. Log tiền xử lý ghi vào `logs/`.

**Nên dùng cache có sẵn (mặc định)**, trừ khi cố ý đổi model OCR/VLM. Dùng chung đúng file cache với nhóm thì kết quả server và Kaggle mới so sánh được với nhau. `text_preprocessed.json` có `meta.pipeline_key`: nếu cấu hình text khác thì code tự bỏ qua cache và tạo lại, nên không lo dùng nhầm. Nếu sinh lại cache trên server, ghi rõ model đã dùng (`meta` của `image_preprocessed.json`) và upload lên Kaggle dataset (mục 7.6).

### 7.5. CoT distillation trên server

Thứ tự và cổng quyết định theo `docs/cot_distillation_plan.md`; lệnh đầy đủ có trong docstring đầu mỗi script. `<R>` là `data/server/cache/rationales`.

| Giai đoạn | Lệnh chính |
|---|---|
| 0 | `run_experiments.py --plan configs/experiments/cot_distill_stage0.yaml` (cột a/b, phẳng / phân cấp / cascade × 3 seed) |
| 1 | `make_id_lists.py`; `generate_rationales.py --split train --ids-file <R>/ids/pilot_dev.txt --k 1 --temperature 0`, rồi `pilot_eval.txt --k 3`; `generate_rationales.py --split public_test --ids-file <R>/ids/val.txt --k 1 --temperature 0`; `eval_teacher.py --raw-dir <R>/raw/v1 ...` |
| 2 | `generate_rationales.py --split train --ids-file <R>/ids/batch{1,2,3}.txt --k {5,3,3} --num-shards N --shard i` |
| 3 | `build_rationale_dataset.py group`, `build_rationale_dataset.py rationalize` (rồi `generate_rationales.py --with-label`), `build_rationale_dataset.py export` |
| 4 | chấm `<R>/review/*.csv`; `build_rationale_dataset.py kappa --a ... --b ...` |
| 5 | `build_rationale_dataset.py merge --reviews <R>/review/*.csv`, `embed_rationales.py` |
| 6 | `run_experiments.py --plan .../cot_distill_grid.yaml`, rồi `.../cot_distill.yaml`, `summarize_results.py`; cuối cùng `--extra-eval test` và `summarize_results.py --with-test` |

`build_rationale_dataset.py` nhận `--config` / `--override` **sau** tên lệnh (vd `build_rationale_dataset.py merge --override ... --reviews ...`). Kết quả đánh giá teacher ghi vào `data/server/outputs/cot_distill/teacher_eval/<tên>/`.

### 7.6. Đồng bộ với nhóm

- **Code**: phần chạy trên server nằm ở branch riêng `thanhtruc-main` (không dùng notebook). Định kỳ merge `main` vào để nhận các sửa chung trong `src/` của nhóm. Trên server chỉ `git pull`, không sửa code trực tiếp trên server rồi quên commit.
- **Cache**: khi sinh mới `image_preprocessed.json` hoặc `rationale_dataset.json`, upload thành phiên bản mới của Kaggle dataset:
  ```bash
  kaggle datasets version -p /data/<user>/vimmsd_cache/<thư mục upload> -m "<model/prompt version, commit>"
  ```
  Thành viên chạy Kaggle chỉ cần *Add Input* dataset đó.
- **Kết quả**: với `VIMMSD_ENV=server`, `train.py` ghi thẳng vào `reports/results.md` với cột môi trường là `server`. Thêm loại GPU vào ghi chú (`--note` hoặc dòng plan), vì thời gian train không so được với Kaggle T4.

---

## 8. Môi trường & dependencies

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
py_vncorenlp         # tách từ VnCoreNLP cho PhoBERT (cần Java >= 8)
underthesea          # tách từ dự phòng (word_segment: underthesea)
emoji                # xử lý emoji trong caption
scikit-learn         # metrics
pyyaml               # đọc config
pandas
matplotlib
pillow
tqdm
paddleocr>=3.0       # phát hiện vùng chữ cho OCR ở bước tiền xử lý ảnh
paddlepaddle         # backend cho paddleocr
einops               # cần cho vietocr (và Vintern)
gdown                # cần cho vietocr
kaggle               # download data qua API (chỉ dùng local)
pytest
```

CLIP được load qua `transformers` (`CLIPVisionModel`, checkpoint `openai/clip-vit-base-patch32`), không cần package `clip` của OpenAI. Nhờ vậy lấy được embedding từng patch cho cross-attention, và đổi sang ViT khác chỉ cần sửa tên model trong config.

`requirements-kaggle.txt` (dùng trong cell bootstrap). Kaggle/Colab đã cài sẵn `torch`, `torchvision`, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`. **Không cài lại `torch`** vì dễ làm lệch phiên bản CUDA của môi trường:

```
py_vncorenlp
underthesea
emoji
```

Tách từ mặc định dùng VnCoreNLP (công cụ PhoBERT dùng khi pretrain), cần Java. Model tách từ tự tải về `~/.cache/vncorenlp` (đổi bằng biến môi trường `VNCORENLP_DIR`) ở lần chạy đầu.

OCR gồm 2 bước: PaddleOCR phát hiện vùng chữ, VietOCR nhận dạng từng dòng (model nhận dạng của PaddleOCR thiếu các chữ cái mang dấu thanh tiếng Việt). `vietocr` ghim `pillow`/`einops`/`gdown` bản cũ nên không nằm trong `requirements.txt`, cài riêng bằng `pip install --no-deps vietocr`.

`paddleocr` nặng và chỉ cần khi tạo cache ảnh → text (chạy 1 lần) nên không nằm trong file này. `notebooks/01b_image_text_extraction.ipynb` tự cài `paddlepaddle-gpu paddleocr timm einops gdown` và `vietocr` (`timm`, `einops` cần cho Vintern). Các notebook train chỉ đọc cache JSON nên không cần cài OCR/VLM.

Không cần môi trường ảo phức tạp. Kaggle/Colab cài qua cell bootstrap, còn local thì chạy `pip install -r requirements.txt && pip install -e .` một lần. Kiểm tra nhanh (không cần GPU/mạng): `pytest tests`.

Trên GPU server dùng `venv` (mục 7.1), cài `torch` theo đúng bản CUDA của driver, rồi cài `requirements.txt`. Trên server `paddlepaddle` nên thay bằng `paddlepaddle-gpu` nếu chạy OCR. Nếu sinh reasoning chậm với `transformers`, có thể cài thêm `vllm` để chạy teacher (Qwen2.5-VL) với batch lớn. Đây chỉ là tuỳ chọn, chưa có trong code.

---

## 9. Bước tiếp theo

Khung code ở mục 2 đã có đủ (trừ `PLAN.md`). Việc cần làm để chạy thật:

1. ~~`git init`, push lên GitHub, sửa `REPO` trong cell bootstrap~~ (đã xong: repo public [yunaLee21/Multimodal_Sarcasm_Detection](https://github.com/yunaLee21/Multimodal_Sarcasm_Detection), không cần `GITHUB_TOKEN`).
2. Chạy `notebooks/00_dataset_check.ipynb` trên Kaggle (attach `phuclkz/vimmsd-private-dataset`), đối chiếu đường dẫn nó in ra với `paths.kaggle.data_dir` trong `configs/base.yaml`.
3. Xử lý các điểm "CHÚ Ý" trong bảng tổng kết của notebook (tiền xử lý caption, cách chia val/test).
4. Chạy `notebooks/00_template.ipynb` trên Kaggle để chắc chắn clone + import `vimmsd` + đọc dữ liệu hoạt động, rồi chạy `01_eda.ipynb` và `01b_image_text_extraction.ipynb` (tạo cache OCR + mô tả VLM, cần GPU).
5. Sau đó các thành viên làm song song theo notebook 02–07, chỉnh hyperparameter qua config/`--override`, sửa code trong `src/` qua PR.
6. GPU server (CoT distillation): code cho mọi giai đoạn đã có (mục 2, mục 7.5); việc còn lại là chạy theo `docs/cot_distillation_plan.md` → `build_rationale_dataset.py` → `embed_rationales.py`.
