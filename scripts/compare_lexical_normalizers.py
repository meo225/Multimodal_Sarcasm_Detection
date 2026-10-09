"""So sánh bảng tra teencode của dự án với model BARTpho ViLexNorm trên caption ViMMSD (chạy trên GPU, ví dụ Kaggle).

    python scripts/compare_lexical_normalizers.py data/raw/vimmsd-train.json --limit 500

Hai bên nhận CÙNG một input: caption đã làm sạch bằng `clean_text` nhưng chưa thay teencode và chưa tách từ.
Như vậy cột `khac_lookup` chỉ phản ánh khác biệt về chuẩn hóa từ, không lẫn với các bước làm sạch khác
(xóa URL, bỏ dấu #, rút gọn chữ lặp...). So sánh không phân biệt hoa thường và khoảng trắng.
Không có nhãn đúng: "khác" không cho biết bên nào đúng, cần đọc tay các dòng khác nhau trong CSV."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from vimmsd.data.preprocessing import SPACE_RE, clean_text, normalize_teencode

MODEL_ID = "duckling2211/bartpho-teencode-vilexnorm"  # BARTpho fine-tune trên ViLexNorm, license CC BY-NC-SA 4.0


def load_captions(path, limit):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = list(payload.values()) if isinstance(payload, dict) else payload
    records = records[:limit] if limit else records
    return [str(r.get("caption", "")) for r in records]


def _canon(text):
    return SPACE_RE.sub(" ", text).strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="file JSON của ViMMSD")
    parser.add_argument("--limit", type=int, default=100, help="số caption đầu tiên, 0 = tất cả")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256, help="số token input tối đa của BARTpho")
    parser.add_argument("--output", type=Path, default=Path("outputs/lexical_comparison.csv"))
    args = parser.parse_args()

    captions = load_captions(args.input, args.limit)
    inputs = [clean_text(c, normalize_teencode_=False, emoji_mode="keep", word_segment_="none") for c in captions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device).eval()

    outputs, truncated = [], []
    for i in tqdm(range(0, len(inputs), args.batch_size), desc="BARTpho"):
        batch = inputs[i:i + args.batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        truncated += [len(ids) > args.max_length for ids in tokenizer(batch)["input_ids"]]
        enc = enc.to(device)
        with torch.inference_mode():
            # chuẩn hóa từ gần như giữ nguyên độ dài: giới hạn output theo độ dài input thay vì luôn 256 token
            generated = model.generate(**enc, num_beams=4, max_new_tokens=enc["input_ids"].shape[1] + 16)
        outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)

    result = pd.DataFrame({
        "caption_goc": captions,
        "input_chung": inputs,
        "lookup": [normalize_teencode(t) for t in inputs],
        "bartpho_vilexnorm": outputs,
        "bi_cat": truncated,  # input dài hơn --max-length: output của BARTpho thiếu phần cuối
    })
    result["khac_lookup"] = [_canon(a) != _canon(b) for a, b in zip(result["lookup"], result["bartpho_vilexnorm"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(result)} dòng vào {args.output}")
    print(f"BARTpho khác bảng tra ở {int(result['khac_lookup'].sum())}/{len(result)} caption "
          f"({int(result['bi_cat'].sum())} caption bị cắt do quá dài)")


if __name__ == "__main__":
    main()
