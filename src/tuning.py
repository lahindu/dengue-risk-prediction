# Step 4 - hyperparameter tuning
# GridSearchCV with TimeSeriesSplit on the training years only (2007-2024).
# TimeSeriesSplit always trains on earlier years and checks on later years.
#
# run:  python -m src.tuning    (slow, about 6 minutes)

import os
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import FEATURES, TARGET, REPORTS_FOLDER
from src.data_prep import prepare_data
from src.models import make_pipeline, random_forest_model, xgboost_model, lightgbm_model, catboost_model

# "model__" means the parameter is inside the "model" step of the pipeline
grids = {
    "Random Forest": (random_forest_model, {
        "model__max_depth": [10, 20],
        "model__min_samples_leaf": [3, 10],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": [None, "balanced"],
    }),
    "XGBoost": (xgboost_model, {
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1],
        "model__scale_pos_weight": [1, 3],
    }),
    "LightGBM": (lightgbm_model, {
        "model__num_leaves": [7, 15, 31],
        "model__learning_rate": [0.05, 0.1],
        "model__scale_pos_weight": [1, 3],
    }),
    "CatBoost": (catboost_model, {
        "model__depth": [4, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__auto_class_weights": ["SqrtBalanced", "Balanced"],
    }),
}


def main():
    train, test, inference, thresholds, district_list = prepare_data(show_info=False)
    train = train.sort_values("start_date")   # TimeSeriesSplit needs the rows in time order
    X = train[FEATURES]
    y = train[TARGET]

    all_results = []
    for name in grids:
        make_model, grid = grids[name]
        search = GridSearchCV(make_pipeline(make_model()), grid, cv=TimeSeriesSplit(n_splits=3),
                              scoring={"f1": "f1", "pr_auc": "average_precision"}, refit="f1")
        start = time.time()
        search.fit(X, y)
        print(f"{name}: {len(search.cv_results_['params'])} combinations, {time.time() - start:.0f}s")
        print("  best F1:", round(search.best_score_, 3))
        print("  best parameters:", search.best_params_)

        results = pd.DataFrame(search.cv_results_)[["params", "mean_test_f1", "std_test_f1",
                                                    "mean_test_pr_auc", "rank_test_f1"]]
        results.insert(0, "model", name)
        all_results.append(results.sort_values("rank_test_f1"))

    pd.concat(all_results).to_csv(os.path.join(REPORTS_FOLDER, "tuning_results.csv"), index=False)
    print("saved reports/tuning_results.csv")


if __name__ == "__main__":
    main()
