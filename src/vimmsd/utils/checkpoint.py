from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    # ghi ra file tạm rồi rename: session free tier bị ngắt giữa lúc ghi sẽ không làm hỏng checkpoint cũ
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return {k: v for k, v in state.items() if k not in {"model", "optimizer", "scheduler", "scaler"}}


def resolve_resume_path(resume_from):
    """`resume_from` có thể là file .pt hoặc thư mục (ví dụ output version trước được attach
    làm input trên Kaggle) — với thư mục thì lấy last.pt bên trong, tìm đệ quy."""
    if not resume_from:
        return None
    p = Path(resume_from)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.rglob("last.pt"), key=lambda f: f.stat().st_mtime)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"không tìm thấy checkpoint tại {resume_from}")
