# Dengue Risk Prediction - Sri Lanka

Advanced Machine Learning - Final Assignment
Name: [Your Name]  |  Student ID: [Student ID]

## What this project does

Dengue is a big problem in Sri Lanka. The Ministry of Health publishes weekly dengue cases for
all 26 RDHS districts. I wanted to see if I can predict **if a district will have a high-risk
week next week**, so health officers could start fogging and clean-up campaigns earlier.

- **Type:** classification (1 = high risk next week, 0 = normal)
- **High risk** = next week's cases are above the 75th percentile of that district
  (Colombo: more than 224.5 cases, Mannar: more than 3). Each district has its own threshold
  because Colombo has many more cases than small districts.
- **Data:** Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health
  (26,311 district-weeks, Dec 2006 - May 2026), collected in the denguedatahub project.

## Results (test years 2025-2026)

I compared 9 models. The best one on the 2024 validation year was **Stacking**
(Random Forest + XGBoost + LightGBM + CatBoost, combined by Logistic Regression).

| Model | PR-AUC | Recall | F1 | Missed high-risk weeks |
|---|---|---|---|---|
| Persistence rule (no ML) | 0.791 | 0.745 | 0.748 | 215 |
| Logistic Regression | 0.790 | 0.916 | 0.766 | 71 |
| Decision Tree | 0.819 | 0.915 | 0.762 | 72 |
| Random Forest | 0.859 | 0.871 | 0.778 | 109 |
| XGBoost | 0.847 | 0.892 | 0.770 | 91 |
| LightGBM | 0.841 | 0.890 | 0.763 | 93 |
| CatBoost | 0.844 | 0.815 | 0.767 | 156 |
| Soft Voting | 0.853 | 0.861 | 0.765 | 117 |
| **Stacking (chosen)** | **0.858** | **0.900** | **0.773** | **84** |

The "persistence rule" is what someone would do without ML: "if this week is already high,
next week will be high". Stacking finds **131 more high-risk weeks** than this rule.
Full explanation in [reports/RESULTS.md](reports/RESULTS.md).

## How to run

Python 3.12

```
pip install -r requirements.txt
python run_all.py              # EDA, training, charts, SHAP/LIME, inference (about 8 min)
streamlit run app/app.py       # the app -> http://localhost:8501
```

Other steps you can run alone:

```
python -m src.tuning           # hyperparameter search (about 6 min)
python -m src.feature_test     # extra experiments
```

There is also a notebook with everything: `notebooks/Dengue_Risk_Final_Assignment.ipynb`

## Files

```
data/                    the dataset + download_data.py
src/config.py            settings (features, years, folders)
src/data_prep.py         cleaning, missing week, features, target, time split
src/eda.py               exploratory data analysis charts
src/models.py            the 9 models (preprocessing + model pipelines)
src/tuning.py            GridSearchCV with TimeSeriesSplit
src/train.py             train, compare, pick the best model, save .pkl files
src/charts.py            comparison charts
src/explain.py           SHAP and LIME
src/predict.py           inference on the left out week
src/feature_test.py      extra experiments
app/app.py               Streamlit app
reports/                 results, charts, RESULTS.md
docs/                    parameters, front end prompt, video plan, viva notes
run_all.py               runs everything
```

## Assignment checklist

| Requirement | Where |
|---|---|
| Local Sri Lankan problem and data | `data/`, this README |
| EDA | `src/eda.py`, `reports/figures/eda_*.png` |
| Preprocessing | `src/data_prep.py` (missing week, features), `src/models.py` (imputation, log, scaling, one-hot) |
| More than RF and XGBoost | 9 models incl. LightGBM, CatBoost, Soft Voting, Stacking |
| Compare with existing approach | Persistence rule baseline |
| Hyperparameter tuning | `src/tuning.py`, `docs/PARAMETERS.md` |
| Evaluation and why the model is suitable | `reports/RESULTS.md` |
| SHAP and LIME | `src/explain.py`, `reports/figures/xai_*.png` |
| App with user input -> prediction | `app/app.py` |
| Saved model (pickle) | `models/model.pkl` |

## Use of AI tools

The assignment allows AI tools. I used ChatGPT/Gemini to help with the Streamlit app
(prompt in `docs/FRONTEND_PROMPT.md`) and to fix some errors in my code. The problem, the
features, the model choices, the parameters and the interpretation are my own work.
