import copy

import numpy as np
from torch.utils.data import DataLoader

from vimmsd.data.dataset import ViMMSDDataset
from vimmsd.training.trainer import predict


def derangement(n, rng):
    """Hoán vị không có điểm bất động: đảm bảo mẫu nào cũng nhận ảnh của mẫu khác."""
    if n < 2:
        raise ValueError("cần ít nhất 2 mẫu để tráo ảnh")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def swap_images(records, seed=0, image_pool=None):
    """Giữ nguyên text, thay ảnh của từng mẫu bằng ảnh của mẫu khác.
    image_pool=None: tráo ảnh trong chính tập records; nếu truyền list records khác
    (ví dụ toàn bộ mẫu not-sarcasm) thì lấy ảnh ngẫu nhiên từ pool đó."""
    rng = np.random.default_rng(seed)
    swapped = copy.deepcopy(records)
    if image_pool is None:
        perm = derangement(len(records), rng)
        for rec, j in zip(swapped, perm):
            rec["image_path"] = records[j]["image_path"]
    else:
        for rec in swapped:
            choices = [r for r in image_pool if r["image_path"] != rec["image_path"]]
            rec["image_path"] = choices[rng.integers(len(choices))]["image_path"]
    return swapped


def run_shortcut_check(model, records, collator, labels, device, target_labels=("multi-sarcasm", "image-sarcasm"),
                       text_preprocessor=None, image_pool=None, batch_size=32, seed=0, use_amp=False, cache_dir=None):
    """Nếu model thực sự dùng ảnh thì dự đoán cho các mẫu sarcasm liên quan tới ảnh phải đổi khi tráo ảnh.
    flip_rate thấp -> model chủ yếu dựa vào text (shortcut learning)."""
    target_ids = {labels.index(l) for l in target_labels}
    subset = [r for r in records if r["label"] in target_ids]
    swapped = swap_images(subset, seed=seed, image_pool=image_pool)

    def run(recs):
        ds = ViMMSDDataset(recs, text_preprocessor=text_preprocessor, cache_dir=cache_dir)
        return predict(model, DataLoader(ds, batch_size=batch_size, collate_fn=collator), device, use_amp)

    orig, swap = run(subset), run(swapped)
    y = np.array(orig["labels"])
    p_orig, p_swap = np.array(orig["preds"]), np.array(swap["preds"])
    changed = p_orig != p_swap
    # total variation distance giữa 2 phân phối xác suất, 0 = không đổi, 1 = đổi hoàn toàn
    tv = 0.5 * (orig["probs"] - swap["probs"]).abs().sum(-1).numpy()

    per_class = {}
    for cid in sorted(target_ids):
        m = y == cid
        per_class[labels[cid]] = {
            "n": int(m.sum()),
            "flip_rate": float(changed[m].mean()),
            "acc_original": float((p_orig[m] == cid).mean()),
            "acc_swapped": float((p_swap[m] == cid).mean()),
            "mean_prob_shift": float(tv[m].mean()),
        }
    return {
        "n": len(subset),
        "flip_rate": float(changed.mean()),
        "acc_original": float((p_orig == y).mean()),
        "acc_swapped": float((p_swap == y).mean()),
        "mean_prob_shift": float(tv.mean()),
        "per_class": per_class,
        "ids": orig["ids"],
        "preds_original": p_orig.tolist(),
        "preds_swapped": p_swap.tolist(),
    }
