import os
import subprocess
import sys


def detect_env() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def git_commit(short: bool = True) -> str:
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
