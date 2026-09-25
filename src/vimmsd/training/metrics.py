import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels):
    idx = list(range(len(labels)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="weighted", zero_division=0
    )[2]
    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def format_metrics(metrics):
    lines = [
        f"macro-F1 {metrics['macro_f1']:.4f} | macro-P {metrics['macro_precision']:.4f} | "
        f"macro-R {metrics['macro_recall']:.4f} | acc {metrics['accuracy']:.4f}"
    ]
    for name, m in metrics["per_class"].items():
        lines.append(f"  {name:<15} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")
    return "\n".join(lines)


def plot_confusion_matrix(cm, labels, path=None, normalize=True, title=None):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{shown[i, j]:.2f}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > shown.max() / 2 else "black")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
