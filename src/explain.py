# Step 6 - explain the best model with SHAP and LIME
#
# run after train.py:  python -m src.explain    (about 2-3 minutes)
#
# Both SHAP and LIME explain the probability of "high risk" from the whole pipeline,
# so their numbers can be compared.

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

from src.config import FEATURES, TARGET, FEATURE_NAMES, MODELS_FOLDER, FIGURES_FOLDER, XAI_FOLDER, RANDOM_STATE
from src.data_prep import prepare_data
from src.train import load_pickle


def save_chart(file_name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, file_name), dpi=130, bbox_inches="tight")
    plt.close("all")


# ------------------------------------------------------------------
# helpers (also used by the app)
# ------------------------------------------------------------------
def make_predict_function(model):
    # SHAP and LIME give numpy arrays, but my pipeline needs a DataFrame with column names
    def predict(values):
        rows = pd.DataFrame(values, columns=FEATURES)
        return model.predict_proba(rows)[:, 1]
    return predict


def make_shap_explainer(model, background_rows, size=100):
    # background = normal training weeks. SHAP compares each prediction with these
    background = background_rows[FEATURES].sample(min(size, len(background_rows)), random_state=RANDOM_STATE)
    masker = shap.maskers.Independent(background.values)
    return shap.PermutationExplainer(make_predict_function(model), masker, feature_names=FEATURES,
                                     seed=RANDOM_STATE)


def get_shap_values(explainer, rows):
    # max_evals = how many times the model is called for each row (more = more exact, slower)
    return explainer(rows[FEATURES].values.astype(float), max_evals=4 * len(FEATURES) + 1, silent=True)


def shap_table(shap_row, row, district_list):
    # one prediction as a small table: feature, value, SHAP
    table = []
    for i in range(len(FEATURES)):
        feature = FEATURES[i]
        value = row[feature].iloc[0]
        if feature == "district_code":
            value = district_list[int(value)]
        else:
            value = round(float(value), 2)
        table.append({"feature": FEATURE_NAMES[feature], "value": str(value),
                      "shap": round(float(shap_row.values[i]), 4)})
    table = pd.DataFrame(table)
    return table.reindex(table["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def make_lime_explainer(train, district_list, seed=RANDOM_STATE):
    district_index = FEATURES.index("district_code")
    return LimeTabularExplainer(
        training_data=train[FEATURES].values.astype(float),
        feature_names=FEATURES,
        class_names=["Normal", "High risk"],
        categorical_features=[district_index],
        categorical_names={district_index: district_list},
        mode="classification",
        random_state=seed,
    )


def lime_explain(lime_explainer, model, row, num_features=5, num_samples=2000):
    predict = make_predict_function(model)

    def predict_two_columns(values):
        p = predict(values)
        return np.column_stack([1 - p, p])

    return lime_explainer.explain_instance(row[FEATURES].values[0].astype(float), predict_two_columns,
                                           num_features=num_features, num_samples=num_samples, labels=(1,))


def lime_top_features(explanation, k):
    # LIME gives (feature index, weight) sorted by importance
    pairs = explanation.as_map()[1]
    top = []
    for index, weight in pairs[:k]:
        top.append(FEATURES[index])
    return top


# ------------------------------------------------------------------
# the actual study
# ------------------------------------------------------------------
def main():
    train, test, inference, thresholds, district_list = prepare_data(show_info=False)
    model = load_pickle(os.path.join(MODELS_FOLDER, "model.pkl"))
    with open(os.path.join(MODELS_FOLDER, "model_info.json")) as f:
        best_model = json.load(f)["best_model"]
    print("Explaining:", best_model)

    labels = [FEATURE_NAMES[f] for f in FEATURES]
    colombo = inference[inference["district"] == "Colombo"]   # held-out week, never seen

    # ---------- 1. SHAP global (300 test weeks) ----------
    explainer = make_shap_explainer(model, train)
    sample = test.sample(300, random_state=RANDOM_STATE)
    values = get_shap_values(explainer, sample)
    values.feature_names = labels

    shap.plots.beeswarm(values, max_display=len(FEATURES), show=False)
    plt.title("SHAP - effect of each feature on P(high risk), 300 test weeks")
    save_chart("xai_shap_beeswarm.png")
    shap.plots.bar(values, max_display=len(FEATURES), show=False)
    plt.title("SHAP global importance (mean |SHAP|)")
    save_chart("xai_shap_bar.png")

    shap_importance = pd.Series(np.abs(values.values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print("\nSHAP global ranking:")
    print(shap_importance.round(4))

    # ---------- 2. SHAP local (Colombo held-out week) ----------
    colombo_values = explainer(colombo[FEATURES].values.astype(float), max_evals=10 * len(FEATURES) + 1,
                               silent=True)
    colombo_values.feature_names = labels
    shap.plots.waterfall(colombo_values[0], max_display=len(FEATURES), show=False)
    plt.title("SHAP - Colombo, week of 2026-05-04 (held out)")
    save_chart("xai_shap_waterfall_colombo.png")
    colombo_table = shap_table(colombo_values[0], colombo, district_list)
    colombo_table.to_csv(os.path.join(XAI_FOLDER, "shap_colombo.csv"), index=False)
    probability = model.predict_proba(colombo[FEATURES])[0, 1]
    print(f"\nColombo: probability {probability:.2f}, SHAP base value {colombo_values.base_values[0]:.3f}")
    print(colombo_table)

    # ---------- 3. LIME local (same week) ----------
    lime_explainer = make_lime_explainer(train, district_list)
    exp = lime_explain(lime_explainer, model, colombo, num_features=len(FEATURES), num_samples=5000)
    exp.as_pyplot_figure(label=1).set_size_inches(9, 5)
    plt.title("LIME - Colombo, week of 2026-05-04")
    save_chart("xai_lime_colombo.png")
    lime_table = pd.DataFrame(exp.as_list(label=1), columns=["condition", "weight"])
    lime_table.to_csv(os.path.join(XAI_FOLDER, "lime_colombo.csv"), index=False)
    print("\nLIME for Colombo (R2 of the local model:", round(exp.score, 3), ")")
    print(lime_table)

    # ---------- 4. LIME stability + SHAP vs LIME on 10 test weeks ----------
    # 5 weeks the model got right and 5 it got wrong.
    # LIME is random, so I run it with 10 different seeds and count how often each
    # feature ends up in the top 5.
    predictions = model.predict(test[FEATURES])
    correct = test[predictions == test[TARGET]].sample(5, random_state=RANDOM_STATE)
    wrong = test[predictions != test[TARGET]].sample(5, random_state=RANDOM_STATE)
    cases = pd.concat([correct.assign(result="correct"), wrong.assign(result="wrong")])

    top5_count = pd.Series(0.0, index=FEATURES)
    rows = []
    for i in range(len(cases)):
        case = cases.iloc[[i]]
        top5_sets = []
        for seed in range(10):
            e = lime_explain(make_lime_explainer(train, district_list, seed), model, case)
            top5 = lime_top_features(e, 5)
            top5_sets.append(set(top5))
            for feature in top5:
                top5_count[feature] += 1

        # Jaccard = overlap between two top-5 lists (1 = exactly the same features)
        jaccards = []
        for a in range(10):
            for b in range(a + 1, 10):
                jaccards.append(len(top5_sets[a] & top5_sets[b]) / len(top5_sets[a] | top5_sets[b]))

        case_shap = explainer(case[FEATURES].values.astype(float), max_evals=10 * len(FEATURES) + 1, silent=True)
        shap_top3 = list(pd.Series(np.abs(case_shap.values[0]), index=FEATURES).sort_values(ascending=False).index[:3])
        lime_top3 = lime_top_features(lime_explain(lime_explainer, model, case), 3)

        rows.append({
            "district": case["district"].iloc[0],
            "week": str(case["start_date"].iloc[0].date()),
            "actual": int(case[TARGET].iloc[0]),
            "probability": round(float(model.predict_proba(case[FEATURES])[0, 1]), 3),
            "result": case["result"].iloc[0],
            "lime_stability_jaccard": round(np.mean(jaccards), 3),
            "shap_top3": ", ".join(shap_top3),
            "lime_top3": ", ".join(lime_top3),
            "same_features_in_top3": len(set(shap_top3) & set(lime_top3)),
        })
    compare = pd.DataFrame(rows)
    compare.to_csv(os.path.join(XAI_FOLDER, "lime_vs_shap_cases.csv"), index=False)
    print("\nSHAP vs LIME on 10 test weeks:")
    print(compare.to_string(index=False))

    top5_share = (top5_count / (len(cases) * 10)).sort_values(ascending=False)
    top5_share.round(3).to_csv(os.path.join(XAI_FOLDER, "lime_top5_share.csv"), header=["share_of_runs_in_top5"])
    plt.figure(figsize=(8, 4.5))
    plt.barh([FEATURE_NAMES[f] for f in top5_share.index[::-1]], top5_share.values[::-1], color="teal")
    plt.xlim(0, 1)
    plt.xlabel("Share of LIME runs (10 weeks x 10 seeds) where the feature was in the top 5")
    plt.title("LIME stability")
    save_chart("xai_lime_stability.png")

    # ---------- 5. permutation importance (a third method to check SHAP) ----------
    # shuffle one column at a time and see how much the test PR-AUC drops
    result = permutation_importance(model, test[FEATURES], test[TARGET], scoring="average_precision",
                                    n_repeats=5, random_state=RANDOM_STATE)
    perm = pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)
    perm.round(4).to_csv(os.path.join(XAI_FOLDER, "permutation_importance.csv"), header=["PR_AUC_drop"])
    plt.figure(figsize=(8, 4.5))
    plt.barh([FEATURE_NAMES[f] for f in perm.index[::-1]], perm.values[::-1], color="darkorange")
    plt.xlabel("Drop in test PR-AUC when the feature is shuffled")
    plt.title("Permutation importance")
    save_chart("xai_permutation_importance.png")
    print("\nPermutation importance:")
    print(perm.round(4))

    # ---------- 6. are wrong predictions close to the threshold? ----------
    right = model.predict(sample[FEATURES]) == sample[TARGET].values
    distance = sample["cases_vs_threshold"].abs()
    print(f"\nMedian distance from the threshold: correct {distance[right].median():.2f}, "
          f"wrong {distance[~right].median():.2f}")

    summary = {
        "model": best_model,
        "colombo_probability": round(float(probability), 3),
        "shap_base_value": round(float(colombo_values.base_values[0]), 3),
        "shap_ranking": list(shap_importance.index),
        "permutation_ranking": list(perm.index),
        "lime_top5_ranking": list(top5_share.index),
        "lime_mean_jaccard": round(float(compare["lime_stability_jaccard"].mean()), 3),
        "shap_lime_mean_overlap_top3": round(float(compare["same_features_in_top3"].mean()), 2),
        "median_distance_correct": round(float(distance[right].median()), 3),
        "median_distance_wrong": round(float(distance[~right].median()), 3),
    }
    with open(os.path.join(XAI_FOLDER, "xai_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved charts in reports/figures and tables in reports/xai")


if __name__ == "__main__":
    main()
