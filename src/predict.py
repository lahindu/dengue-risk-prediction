# Inference on data the model has never seen
# The newest week (2026-05-04) was kept out of training and testing.
# I load the saved .pkl file and predict one sample (Colombo), then all 26 districts.
#
# run after train.py:  python -m src.predict

import os

from src.config import FEATURES, TARGET, MODEL_NAMES, MODELS_FOLDER, REPORTS_FOLDER
from src.data_prep import prepare_data
from src.train import load_pickle, model_file


def label(value):
    if value == 1:
        return "High risk"
    return "Normal"


def main():
    train, test, inference, thresholds, district_list = prepare_data(show_info=False)

    sample = inference[inference["district"] == "Colombo"]
    print("Left out sample: Colombo, week of", sample["start_date"].iloc[0].date())
    for feature in FEATURES:
        print(f"  {feature:24s} {round(float(sample[feature].iloc[0]), 3)}")

    model = load_pickle(os.path.join(MODELS_FOLDER, "model.pkl"))
    probability = model.predict_proba(sample[FEATURES])[0, 1]
    print(f"\nPrediction (models/model.pkl): {label(model.predict(sample[FEATURES])[0])} "
          f"({probability:.1%} chance of high risk)")
    print(f"Real answer: {int(sample['cases_next_week'].iloc[0])} cases next week, threshold "
          f"{sample['risk_threshold'].iloc[0]} -> {label(sample[TARGET].iloc[0])}")

    print("\nSame sample with every model:")
    for name in MODEL_NAMES:
        m = load_pickle(model_file(name))
        p = m.predict_proba(sample[FEATURES])[0, 1]
        print(f"  {name:20s} {label(m.predict(sample[FEATURES])[0]):10s} {p:.1%}")

    print("\nAll 26 districts of the left out week:")
    table = inference[["district", "cases_this_week", "cases_next_week", "risk_threshold", TARGET]].copy()
    table = table.rename(columns={TARGET: "actual"})
    for name in MODEL_NAMES:
        m = load_pickle(model_file(name))
        table[name] = m.predict(inference[FEATURES])
        print(f"  {name:20s} {(table[name] == table['actual']).sum()} / {len(table)} correct")
    table.to_csv(os.path.join(REPORTS_FOLDER, "inference_results.csv"), index=False)


if __name__ == "__main__":
    main()
