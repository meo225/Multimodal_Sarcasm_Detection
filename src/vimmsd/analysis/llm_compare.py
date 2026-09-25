import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from vimmsd.data.image_text import load_vintern, vintern_pixel_values
from vimmsd.training.metrics import compute_metrics

LABEL_DESCRIPTIONS = {
    "not-sarcasm": "bài đăng không mỉa mai",
    "text-sarcasm": "sự mỉa mai thể hiện ngay trong văn bản, không cần nhìn ảnh",
    "image-sarcasm": "sự mỉa mai đến từ mâu thuẫn giữa nội dung ảnh và văn bản",
    "multi-sarcasm": "phải kết hợp cả ảnh và văn bản mới nhận ra được sự mỉa mai",
}

INSTRUCTION = (
    "Bạn là chuyên gia phân tích mạng xã hội Việt Nam. Cho một bài đăng gồm một ảnh và caption, "
    "hãy phân loại bài đăng vào đúng 1 trong 4 nhãn sau:\n{label_list}\n"
    "Chỉ trả lời đúng tên nhãn, không giải thích."
)


def build_instruction(labels):
    label_list = "\n".join(f"- {l}: {LABEL_DESCRIPTIONS.get(l, l)}" for l in labels)
    return INSTRUCTION.format(label_list=label_list)


def build_question(caption):
    return f'Caption: "{caption}"\nNhãn:'


def parse_label(output, labels):
    """Tìm nhãn trong câu trả lời của LLM. Trả về None nếu không tìm được (tính là dự đoán sai)."""
    text = output.lower()
    found = [(text.find(l), l) for l in labels if l in text]
    if found:
        return min(found)[1]  # nhãn xuất hiện đầu tiên
    normalized = re.sub(r"[\s_]+", "-", text)
    for l in labels:
        if l in normalized:
            return l
    return None


class HFVisionLanguageClassifier:
    """LLM đa modal trên HuggingFace dùng chat template chuẩn (Qwen2-VL, Qwen2.5-VL, ...)."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", labels=None, max_new_tokens=10,
                 dtype=torch.bfloat16, device_map="auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        ).eval()
        self.labels = labels
        self.max_new_tokens = max_new_tokens

    def _messages(self, caption, examples):
        messages = [{"role": "system", "content": [{"type": "text", "text": build_instruction(self.labels)}]}]
        images = []
        for ex in examples:  # few-shot: mỗi ví dụ là 1 lượt hỏi-đáp trước câu hỏi thật
            messages.append({"role": "user", "content": [{"type": "image"},
                                                         {"type": "text", "text": build_question(ex["caption"])}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["label_name"]}]})
            images.append(Image.open(ex["image_path"]).convert("RGB"))
        messages.append({"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": build_question(caption)}]})
        return messages, images

    @torch.no_grad()
    def __call__(self, image_path, caption, examples=()):
        messages, images = self._messages(caption, examples)
        images.append(Image.open(image_path).convert("RGB"))
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class VinternClassifier:
    """Vintern (5CD-AI, dựa trên InternVL, tối ưu cho tiếng Việt). Chỉ hỗ trợ zero-shot."""

    def __init__(self, model_name="5CD-AI/Vintern-1B-v3_5", labels=None, max_new_tokens=10):
        self.model, self.tokenizer = load_vintern(model_name)
        self.labels = labels
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def __call__(self, image_path, caption, examples=()):
        if examples:
            raise NotImplementedError("VinternClassifier chỉ hỗ trợ zero-shot")
        question = "<image>\n" + build_instruction(self.labels) + "\n" + build_question(caption)
        return self.model.chat(self.tokenizer, vintern_pixel_values(image_path), question,
                               dict(max_new_tokens=self.max_new_tokens, do_sample=False))


def select_few_shot_examples(records, labels, k_per_class=1, seed=0):
    rng = torch.Generator().manual_seed(seed)
    examples = []
    for cid, name in enumerate(labels):
        pool = [r for r in records if r["label"] == cid]
        for i in torch.randperm(len(pool), generator=rng)[:k_per_class].tolist():
            examples.append({**pool[i], "label_name": name})
    return examples


def evaluate_llm(classifier, records, labels, examples=(), out_path=None):
    """Chạy LLM trên từng mẫu, ghi output thô ra JSONL (chạy tiếp được nếu bị ngắt), trả về metrics."""
    done = {}
    if out_path and Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            done = {row["id"]: row for row in map(json.loads, f)}

    rows = []
    f = open(out_path, "a", encoding="utf-8") if out_path else None
    try:
        for r in tqdm(records, desc="LLM"):
            if r["id"] in done:
                rows.append(done[r["id"]])
                continue
            raw = classifier(r["image_path"], r["caption"], examples)
            pred = parse_label(raw, labels)
            row = {"id": r["id"], "label": labels[r["label"]], "pred": pred, "raw": raw}
            rows.append(row)
            if f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        if f:
            f.close()

    invalid = sum(row["pred"] is None for row in rows)
    # câu trả lời không parse được -> gán nhãn sai chắc chắn để không được tính điểm
    y_true = [labels.index(row["label"]) for row in rows]
    y_pred = [labels.index(row["pred"]) if row["pred"] else (t + 1) % len(labels) for row, t in zip(rows, y_true)]
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics["invalid_rate"] = invalid / max(len(rows), 1)
    return metrics, rows


def compare_predictions(llm_rows, model_out, labels):
    """Ghép dự đoán của LLM và model fine-tune theo id, đánh dấu trường hợp chỉ 1 bên đúng
    để phân tích định tính (LLM hiểu ngữ cảnh văn hóa tốt hơn hay model fine-tune vượt trội)."""
    model_pred = {i: labels[p] for i, p in zip(model_out["ids"], model_out["preds"])}
    df = pd.DataFrame(llm_rows).rename(columns={"pred": "llm_pred"})
    df["model_pred"] = df["id"].map(model_pred)
    df = df.dropna(subset=["model_pred"])
    df["llm_correct"] = df["llm_pred"] == df["label"]
    df["model_correct"] = df["model_pred"] == df["label"]
    df["case"] = "both_wrong"
    df.loc[df.llm_correct & df.model_correct, "case"] = "both_correct"
    df.loc[df.llm_correct & ~df.model_correct, "case"] = "only_llm"
    df.loc[~df.llm_correct & df.model_correct, "case"] = "only_model"
    return df
