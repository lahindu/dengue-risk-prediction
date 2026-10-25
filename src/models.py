# Step 3 and 4 - the models
#
# Every ML model is a Pipeline: preprocessing -> model
# so the same preprocessing is used for training, testing and the app.
# I chose all the parameter values myself (no defaults). The tuning is in tuning.py
# and the reasons are in docs/PARAMETERS.md

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import COUNT_COLUMNS, OTHER_COLUMNS, RANDOM_STATE, RISK_PERCENTILE


def make_preprocessor(one_hot_district=False):
    # case counts: fill empty values with median -> log(1 + x) -> scale
    counts = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # other number columns: fill with median -> scale
    others = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # district: trees can use the code directly, logistic regression needs one-hot
    if one_hot_district:
        district = OneHotEncoder(handle_unknown="ignore")
    else:
        district = "passthrough"

    return ColumnTransformer([
        ("counts", counts, COUNT_COLUMNS),
        ("others", others, OTHER_COLUMNS),
        ("district", district, ["district_code"]),
    ], verbose_feature_names_out=False)


def make_pipeline(model, one_hot_district=False):
    return Pipeline([
        ("preprocess", make_preprocessor(one_hot_district)),
        ("model", model),
    ])


class PersistenceRule:
    # Not ML. The simple rule a health officer would use:
    # "if this week is already above the district threshold, next week will be high risk too"
    # I need this to show that my models are actually better than doing nothing.

    def fit(self, X, y):
        # threshold of each district from the training data
        self.thresholds = X.groupby("district_code")["cases_this_week"].quantile(RISK_PERCENTILE).to_dict()
        return self

    def predict_proba(self, X):
        threshold = X["district_code"].map(self.thresholds)
        # turn "how far above the threshold" into a 0-1 score (needed for ROC-AUC and PR-AUC)
        gap = np.log1p(X["cases_this_week"].fillna(0)) - np.log1p(threshold)
        score = 1 / (1 + np.exp(-3 * gap))
        score = np.array(score, dtype=float)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def logistic_regression():
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    return make_pipeline(model, one_hot_district=True)


def decision_tree():
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced",
                                   random_state=RANDOM_STATE)
    return make_pipeline(model)


def random_forest_model():
    return RandomForestClassifier(
        n_estimators=300,         # default 100
        max_depth=20,             # default None (no limit)
        min_samples_leaf=10,      # default 1, a leaf needs at least 10 weeks
        max_features="sqrt",      # tested sqrt and 0.5, almost the same, sqrt is faster
        class_weight="balanced",  # default None, only ~24% of weeks are high risk
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def xgboost_model():
    return XGBClassifier(
        n_estimators=400,         # default 100
        max_depth=3,              # default 6, deeper trees overfit in my tests
        learning_rate=0.05,       # default 0.3
        subsample=0.8,            # default 1
        colsample_bytree=0.8,     # default 1
        min_child_weight=3,       # default 1
        scale_pos_weight=3,       # default 1, about 3 normal weeks for each high risk week
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def lightgbm_model():
    return LGBMClassifier(
        n_estimators=400,         # default 100
        learning_rate=0.05,       # default 0.1
        num_leaves=7,             # default 31, small trees overfit less
        min_child_samples=40,     # default 20
        subsample=0.8,            # default 1
        subsample_freq=1,         # needed or subsample does nothing
        colsample_bytree=0.8,     # default 1
        reg_lambda=1.0,           # default 0
        scale_pos_weight=3,       # default 1
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def catboost_model():
    return CatBoostClassifier(
        iterations=500,                     # default 1000
        depth=4,                            # default 6
        learning_rate=0.05,                 # default is automatic
        l2_leaf_reg=5,                      # default 3
        auto_class_weights="SqrtBalanced",  # default None
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )


def four_ensembles():
    return [
        ("random_forest", random_forest_model()),
        ("xgboost", xgboost_model()),
        ("lightgbm", lightgbm_model()),
        ("catboost", catboost_model()),
    ]


def soft_voting():
    # average of the 4 probabilities
    return make_pipeline(VotingClassifier(four_ensembles(), voting="soft"))


def stacking():
    # the 4 models predict first, then logistic regression learns how much to trust each one
    model = StackingClassifier(
        estimators=four_ensembles(),
        final_estimator=LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000),
        cv=5,
    )
    return make_pipeline(model)


def create_model(name):
    if name == "Persistence rule":
        return PersistenceRule()
    elif name == "Logistic Regression":
        return logistic_regression()
    elif name == "Decision Tree":
        return decision_tree()
    elif name == "Random Forest":
        return make_pipeline(random_forest_model())
    elif name == "XGBoost":
        return make_pipeline(xgboost_model())
    elif name == "LightGBM":
        return make_pipeline(lightgbm_model())
    elif name == "CatBoost":
        return make_pipeline(catboost_model())
    elif name == "Soft Voting":
        return soft_voting()
    elif name == "Stacking":
        return stacking()
    else:
        raise ValueError("Unknown model: " + name)
