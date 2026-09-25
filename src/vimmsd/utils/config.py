import copy
import re
from pathlib import Path

import yaml

from vimmsd.utils.env import detect_env


class _Loader(yaml.SafeLoader):
    """PyYAML (YAML 1.1) đọc `1e-5` thành string vì thiếu dấu chấm; loader này đọc thành float."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d[\d_]*)?(?:\.\d*)?[eE][-+]?\d+$"),
    list("-+0123456789."),
)


def _yaml_load(text):
    return yaml.load(text, Loader=_Loader)


class Config(dict):
    """dict cho phép truy cập kiểu thuộc tính: cfg.train.lr thay vì cfg["train"]["lr"]."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        return _to_plain(self)


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_with_inherit(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = _yaml_load(f) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return deep_merge(_load_with_inherit(path.parent / parent), raw)


def _parse_override(item: str):
    key, _, value = item.partition("=")
    if not key or not _:
        raise ValueError(f"override phải có dạng key.sub=value, nhận được: {item!r}")
    return key.split("."), _yaml_load(value)


def find_project_root(start) -> Path:
    # đường dẫn tương đối trong config tính từ gốc repo (thư mục có pyproject.toml), không phụ thuộc cwd
    for p in [Path(start).resolve(), *Path(start).resolve().parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def load_config(path, overrides=None, env=None) -> Config:
    """Đọc YAML (hỗ trợ `inherit: base.yaml`), áp override dạng "train.lr=1e-5",
    rồi thay `paths` bằng nhánh ứng với môi trường đang chạy (local/kaggle/colab)."""
    path = Path(path)
    cfg = _load_with_inherit(path)

    for item in overrides or []:
        keys, value = _parse_override(item)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    env = env or detect_env()
    all_paths = cfg.get("paths", {})
    if env not in all_paths:
        raise KeyError(f"configs thiếu paths cho môi trường '{env}'")
    root = find_project_root(path)
    cfg["paths"] = {k: str(root / v) if not Path(v).is_absolute() else v for k, v in all_paths[env].items()}
    cfg["env"] = env
    cfg["config_path"] = str(path)
    cfg.setdefault("name", path.stem)
    return _to_config(cfg)
