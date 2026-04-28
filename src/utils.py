"""
src/utils.py
Shared helper functions used across all notebooks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#2563EB",    # blue
    "secondary": "#DC2626",  # red
    "accent": "#16A34A",     # green
    "neutral": "#6B7280",    # gray
    "bg": "#F9FAFB",
}


def set_style() -> None:
    """Apply a clean, consistent matplotlib style for all notebooks."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 120,
        }
    )


# Call at import so every notebook that does `from src.utils import *` gets it.
set_style()


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Print and return common regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_summary(y_true: np.ndarray, y_pred: np.ndarray,
                            labels: list[str] | None = None) -> None:
    """Print accuracy + full classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy : {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=labels))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           labels: list[str] | None = None,
                           title: str = "Confusion Matrix") -> None:
    """Plot a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(cm))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Gradient descent visualisation
# ---------------------------------------------------------------------------

def plot_cost_history(cost_history: list[float] | np.ndarray,
                      title: str = "Cost vs. Iteration") -> None:
    """Plot the training loss curve produced by gradient descent."""
    iters = np.arange(1, len(cost_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, cost_history, color=PALETTE["primary"], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J(θ)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def plot_regression_line(X: np.ndarray, y: np.ndarray,
                          theta: np.ndarray,
                          x_label: str = "x", y_label: str = "y",
                          title: str = "Linear Regression Fit") -> None:
    """Scatter plot of data with fitted regression line overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color=PALETTE["neutral"], alpha=0.6, label="Data", zorder=3)

    x_line = np.linspace(X.min(), X.max(), 200)
    # Works for simple (1-feature) linear regression
    y_line = theta[0] + theta[1] * x_line
    ax.plot(x_line, y_line, color=PALETTE["primary"], lw=2, label="Fit")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
