# src/visualize.py
import sys
import os

# Ensure non-interactive backend for saving figures
import matplotlib
matplotlib.use("Agg")

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.features import featurize_dataset


# Create a dedicated figures directory at project root
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_importance(fi: pd.Series, out_path: str):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=fi.values, y=fi.index, orient="h")
    plt.title("Feature importance (basic features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_predictions(out_path: str):
    X_basic_df, X_window_mat, y, meta = featurize_dataset()
    clf = joblib.load(os.path.join(PROJECT_ROOT, "model_rf.joblib"))

    X = np.hstack([X_basic_df.values, X_window_mat])
    probs = clf.predict_proba(X)[:, 1]

    df = pd.DataFrame({"mutation": meta, "label": y, "prob_deleterious": probs})
    print(df)

    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=df,
        x="mutation",
        y="prob_deleterious",
        hue="label",
        palette={0: "green", 1: "red"},
        s=90,
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Mutation")
    plt.ylabel("Predicted probability (deleterious)")
    plt.legend(title="Label", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_and_plot_importance(out_path: str):
    # Recompute feature importance from the trained model
    X_basic_df, X_window_mat, y, meta = featurize_dataset()
    X = np.hstack([X_basic_df.values, X_window_mat])

    clf = joblib.load(os.path.join(PROJECT_ROOT, "model_rf.joblib"))
    importances = clf.feature_importances_[: len(X_basic_df.columns)]
    fi = pd.Series(importances, index=X_basic_df.columns).sort_values(ascending=False)

    plot_importance(fi, out_path)


if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Figures will be saved to: {FIG_DIR}")

    pred_path = os.path.join(FIG_DIR, "fig_predictions.png")
    imp_path = os.path.join(FIG_DIR, "fig_feature_importance.png")

    # Generate both plots
    plot_predictions(pred_path)
    compute_and_plot_importance(imp_path)

    print("Saved:")
    print(f" - {pred_path}")
    print(f" - {imp_path}")
