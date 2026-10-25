# Two extra experiments to check my design choices
#
# run:  python -m src.feature_test   (about 1 minute)
#
# A. Do the "cases vs threshold" features help? (train each model with and without them)
# B. How much better than the simple rule is the model when predicting 1, 2 or 4 weeks ahead?

import os

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

import src.config as config
from src.config import FEATURES, TARGET, REPORTS_FOLDER
from src.data_prep import prepare_data, split_train_valid
from src.models import create_model


def score(name, columns, train, test, target):
    # the preprocessor reads OTHER_COLUMNS from config, so I remove the dropped columns there too
    original = config.OTHER_COLUMNS[:]
    config.OTHER_COLUMNS[:] = [c for c in original if c in columns]
    model = create_model(name)
    model.fit(train[columns], train[target])
    prob = model.predict_proba(test[columns])[:, 1]
    config.OTHER_COLUMNS[:] = original
    return round(average_precision_score(test[target], prob), 3), round(f1_score(test[target], prob > 0.5), 3)


def main():
    train, test, inference, thresholds, district_list = prepare_data(show_info=False)
    fit_part, valid_part = split_train_valid(train)
    rows = []

    print("A. with and without the threshold features (validation 2024)")
    without = [f for f in FEATURES if f not in ["cases_vs_threshold", "avg4_vs_threshold"]]
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
        for setting, columns in [("without", without), ("with", FEATURES)]:
            pr_auc, f1 = score(name, columns, fit_part, valid_part, TARGET)
            rows.append({"experiment": "A features", "model": name, "setting": setting + " threshold features",
                         "PR-AUC": pr_auc, "F1": f1})
            print(f"  {name:20s} {setting:8s} PR-AUC {pr_auc}")

    print("B. predicting further ahead (test 2025-2026)")
    for weeks in [1, 2, 4]:
        tr = train.copy()
        te = test.copy()
        for part in [tr, te]:
            future = part.groupby("district")["cases"].shift(-weeks)
            part["target_ahead"] = (future > part["risk_threshold"]).astype(int)
            part["future"] = future
        tr = tr.dropna(subset=["future"])
        te = te.dropna(subset=["future"])

        rule_pr = round(average_precision_score(te["target_ahead"], te["cases_vs_threshold"]), 3)
        rule_f1 = round(f1_score(te["target_ahead"], te["cases_vs_threshold"] > 0), 3)
        pr_auc, f1 = score("LightGBM", FEATURES, tr, te, "target_ahead")
        rows.append({"experiment": "B horizon", "model": "Persistence rule", "setting": f"{weeks} week(s) ahead",
                     "PR-AUC": rule_pr, "F1": rule_f1})
        rows.append({"experiment": "B horizon", "model": "LightGBM", "setting": f"{weeks} week(s) ahead",
                     "PR-AUC": pr_auc, "F1": f1})
        print(f"  {weeks} week(s) ahead: rule PR-AUC {rule_pr}, LightGBM PR-AUC {pr_auc}")

    pd.DataFrame(rows).to_csv(os.path.join(REPORTS_FOLDER, "feature_test.csv"), index=False)
    print("saved reports/feature_test.csv")


if __name__ == "__main__":
    main()
