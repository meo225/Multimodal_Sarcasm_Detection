"""Compare the project lookup baseline with BARTpho ViLexNorm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from vimmsd.data.preprocessing import clean_text


MODEL_ID = "duckling2211/bartpho-teencode-vilexnorm"


def load_captions(path: Path, limit: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.values()) if isinstance(payload, dict) else payload
    return [str(item.get("caption", "")) for item in records[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a ViMMSD JSON file")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/lexical_comparison.csv"))
    args = parser.parse_args()

    captions = load_captions(args.input, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    encoded = tokenizer(
        captions, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=256, num_beams=4)
    model_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

    result = pd.DataFrame(
        {
            "caption_goc": captions,
            "lookup": [clean_text(text, emoji_mode="keep", word_segment_=False) for text in captions],
            "bartpho_vilexnorm": model_outputs,
        }
    )
    result["khac_lookup"] = result["lookup"] != result["bartpho_vilexnorm"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(result)} dòng vào {args.output}")
    print(f"Model thay đổi {int(result['khac_lookup'].sum())}/{len(result)} caption so với lookup.")


if __name__ == "__main__":
    main()
