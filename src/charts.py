# Step 5 - charts to compare the models
#
# run after train.py:  python -m src.charts

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import FEATURES, TARGET, MODELS_FOLDER, REPORTS_FOLDER, FIGURES_FOLDER, FEATURE_NAMES
from src.data_prep import prepare_data
from src.train import load_pickle, model_file

# colour for each type of model
colours = {
    "Persistence rule": "grey", "Logistic Regression": "grey", "Decision Tree": "grey",
    "Random Forest": "teal",
    "XGBoost": "darkorange", "LightGBM": "darkorange", "CatBoost": "darkorange",
    "Soft Voting": "gold", "Stacking": "gold",
}


def bar_charts(valid_table, test_table, best_model):
    order = test_table.sort_values("PR-AUC").index
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    charts = [("Validation 2024 - PR-AUC (used to choose)", valid_table, "PR-AUC"),
              ("Test 2025-2026 - PR-AUC", test_table, "PR-AUC"),
              ("Test 2025-2026 - F1", test_table, "F1")]
    for i in range(3):
        title, table, metric = charts[i]
        values = table.loc[order, metric]
        ax[i].barh(order, values, color=[colours[m] for m in order])
        for j, v in enumerate(values):
            ax[i].text(v + 0.003, j, f"{v:.3f}", va="center", fontsize=9)
        ax[i].set_xlim(values.min() - 0.1, values.max() + 0.06)
        ax[i].set_title(title)
    fig.suptitle(f"Model comparison - grey: baseline, teal: bagging, orange: boosting, "
                 f"gold: voting/stacking. Best model = {best_model}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "model_comparison.png"), dpi=130)
    plt.close()


def confusion_matrix_charts(matrices):
    fig, ax = plt.subplots(3, 3, figsize=(12, 11))
    ax = ax.flatten()
    for i, name in enumerate(matrices):
        m = matrices[name]
        ax[i].imshow(m, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax[i].text(c, r, m[r][c], ha="center", va="center", fontsize=13)
        ax[i].set_xticks([0, 1], ["Normal", "High risk"])
        ax[i].set_yticks([0, 1], ["Normal", "High risk"])
        ax[i].set_xlabel("Predicted")
        ax[i].set_ylabel("Actual")
        ax[i].set_title(f"{name}\nmissed {m[1][0]}, false alarms {m[0][1]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "confusion_matrices.png"), dpi=130)
    plt.close()


def pr_curves(test, best_model):
    plt.figure(figsize=(8, 6))
    for name in ["Persistence rule", "Logistic Regression", "Random Forest", "XGBoost",
                 "LightGBM", "CatBoost", "Stacking"]:
        model = load_pickle(model_file(name))
        prob = model.predict_proba(test[FEATURES])[:, 1]
        precision, recall, _ = precision_recall_curve(test[TARGET], prob)
        width = 3 if name == best_model else 1.2
        style = "--" if name == "Persistence rule" else "-"
        plt.plot(recall, precision, style, linewidth=width, label=name)
    plt.axhline(test[TARGET].mean(), color="grey", linestyle=":", label="Random guess")
    plt.xlabel("Recall (how many high risk weeks we found)")
    plt.ylabel("Precision (how many alarms were right)")
    plt.title("Precision-recall curves, test years 2025-2026")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "pr_curves.png"), dpi=130)
    plt.close()


def feature_importance():
    # built-in importance of the 4 tree models (scaled so each adds up to 1)
    table = {}
    for name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
        pipe = load_pickle(model_file(name))
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        importance = pd.Series(pipe.named_steps["model"].feature_importances_, index=names, dtype=float)
        table[name] = importance / importance.sum()
    table = pd.DataFrame(table).reindex(FEATURES)
    table.index = [FEATURE_NAMES[f] for f in table.index]
    table.sort_values("LightGBM").plot(kind="barh", figsize=(10, 6))
    plt.title("Built-in feature importance of the tree models")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "feature_importance.png"), dpi=130)
    plt.close()


def main():
    valid_table = pd.read_csv(os.path.join(REPORTS_FOLDER, "validation_metrics.csv"), index_col="Model")
    test_table = pd.read_csv(os.path.join(REPORTS_FOLDER, "metrics.csv"), index_col="Model")
    with open(os.path.join(REPORTS_FOLDER, "confusion_matrices.json")) as f:
        matrices = json.load(f)
    with open(os.path.join(MODELS_FOLDER, "model_info.json")) as f:
        best_model = json.load(f)["best_model"]
    train, test, inference, thresholds, district_list = prepare_data(show_info=False)

    bar_charts(valid_table, test_table, best_model)
    confusion_matrix_charts(matrices)
    pr_curves(test, best_model)
    feature_importance()

    print("Missed high risk weeks and false alarms (test years):")
    for name in matrices:
        print(f"  {name:20s} missed {matrices[name][1][0]:4d}   false alarms {matrices[name][0][1]:4d}")
    print("charts saved in reports/figures")


if __name__ == "__main__":
    main()
