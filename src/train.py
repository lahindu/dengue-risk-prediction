# Step 3 and 5 - train all the models, pick the best one, test it
#
# run:  python -m src.train    (takes about 4 minutes, Stacking is the slow one)
#
# Part A: train on 2007-2023, compare on 2024 -> pick the best model (best PR-AUC)
# Part B: train again on 2007-2024, test once on 2025-2026, save the models as .pkl

import json
import os
import pickle
import time

import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.config import FEATURES, TARGET, MODEL_NAMES, MODELS_FOLDER, REPORTS_FOLDER, RANDOM_STATE
from src.data_prep import prepare_data, split_train_valid
from src.models import create_model, make_pipeline, random_forest_model, xgboost_model, \
    lightgbm_model, catboost_model


def get_scores(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]   # probability of high risk
    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1": f1_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities),
    }
    return scores, confusion_matrix(y, predictions)


def model_file(name):
    # "Random Forest" -> models/random_forest.pkl
    return os.path.join(MODELS_FOLDER, name.lower().replace(" ", "_") + ".pkl")


def save_pickle(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_and_score(name, train, test):
    model = create_model(name)
    start = time.time()
    model.fit(train[FEATURES], train[TARGET])
    seconds = time.time() - start
    scores, matrix = get_scores(model, test[FEATURES], test[TARGET])
    scores["Training time (s)"] = round(seconds, 1)
    print(f"  {name:20s} PR-AUC {scores['PR-AUC']:.3f}  F1 {scores['F1']:.3f}  "
          f"Recall {scores['Recall']:.3f}  ({seconds:.1f}s)")
    return model, scores, matrix


def default_vs_my_parameters(train, test):
    # to show that my parameters are better than the library defaults
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    pairs = [
        ("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), random_forest_model()),
        ("XGBoost", XGBClassifier(random_state=RANDOM_STATE), xgboost_model()),
        ("LightGBM", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1), lightgbm_model()),
        ("CatBoost", CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False),
         catboost_model()),
    ]
    rows = []
    for name, default_model, my_model in pairs:
        for setting, model in [("Default parameters", default_model), ("My parameters", my_model)]:
            pipe = make_pipeline(model)
            pipe.fit(train[FEATURES], train[TARGET])
            scores, _ = get_scores(pipe, test[FEATURES], test[TARGET])
            rows.append({"Model": name, "Setting": setting, "Recall": round(scores["Recall"], 3),
                         "F1": round(scores["F1"], 3), "PR-AUC": round(scores["PR-AUC"], 3)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(REPORTS_FOLDER, "default_vs_tuned.csv"), index=False)
    print(table.to_string(index=False))


def main(compare_defaults=True):
    train, test, inference, thresholds, district_list = prepare_data()
    fit_part, valid_part = split_train_valid(train)

    # ---------- Part A: choose the best model on 2024 ----------
    print("\nPart A - train on 2007-2023, validate on 2024")
    valid_results = {}
    for name in MODEL_NAMES:
        model, scores, matrix = train_and_score(name, fit_part, valid_part)
        valid_results[name] = scores
    valid_table = pd.DataFrame(valid_results).T.round(4)
    valid_table.index.name = "Model"
    valid_table.to_csv(os.path.join(REPORTS_FOLDER, "validation_metrics.csv"))

    best_model = valid_table["PR-AUC"].idxmax()
    print("Best model on 2024 (PR-AUC):", best_model)

    # ---------- Part B: train on all training years and test on 2025-2026 ----------
    print("\nPart B - train on 2007-2024, test on 2025-2026")
    test_results = {}
    matrices = {}
    for name in MODEL_NAMES:
        model, scores, matrix = train_and_score(name, train, test)
        test_results[name] = scores
        matrices[name] = matrix.tolist()
        save_pickle(model, model_file(name))
        if name == best_model:
            save_pickle(model, os.path.join(MODELS_FOLDER, "model.pkl"))   # the one the app uses

    test_table = pd.DataFrame(test_results).T.round(4)
    test_table.index.name = "Model"
    test_table.to_csv(os.path.join(REPORTS_FOLDER, "metrics.csv"))
    with open(os.path.join(REPORTS_FOLDER, "confusion_matrices.json"), "w") as f:
        json.dump(matrices, f, indent=2)

    # info the app needs (district list, thresholds, which model)
    info = {
        "best_model": best_model,
        "validation_metrics": valid_table.loc[best_model].to_dict(),
        "test_metrics": test_table.loc[best_model].to_dict(),
        "district_list": district_list,
        "thresholds": thresholds.round(1).to_dict(),
    }
    with open(os.path.join(MODELS_FOLDER, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # 200 random training rows, SHAP and LIME need some "normal" data to compare with
    train[FEATURES].sample(200, random_state=RANDOM_STATE).to_csv(
        os.path.join(MODELS_FOLDER, "background.csv"), index=False)

    print("\nTest results (2025-2026):")
    print(test_table.drop(columns="Training time (s)").to_string())
    print("\nBest model:", best_model, "-> saved as models/model.pkl")

    if compare_defaults:
        print("\nDefault parameters vs my parameters (test years):")
        default_vs_my_parameters(train, test)


if __name__ == "__main__":
    main()
