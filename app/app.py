# Streamlit app for the dengue risk model
# I generated the first version with ChatGPT/Gemini (prompt in docs/FRONTEND_PROMPT.md)
# and then changed it for my project.
#
# run (after python run_all.py):  streamlit run app/app.py

import json
import os
import pickle
import sys

import pandas as pd
import streamlit as st

# so that "from src..." works when streamlit runs this file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FEATURES, TARGET, MODELS_FOLDER, REPORTS_FOLDER, FIGURES_FOLDER, XAI_FOLDER
from src.data_prep import prepare_data, make_one_row
from src.explain import make_shap_explainer, get_shap_values, shap_table

st.set_page_config(page_title="Dengue Risk Sri Lanka", page_icon="🦟", layout="wide")
st.title("🦟 Dengue Risk Prediction - Sri Lanka")
st.write("Predicts if a district will be **high risk** for dengue **next week**. "
         "Data: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health.")

if not os.path.exists(os.path.join(MODELS_FOLDER, "model.pkl")):
    st.error("No model found. Run `python run_all.py` first.")
    st.stop()


@st.cache_resource
def load_everything():
    with open(os.path.join(MODELS_FOLDER, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_FOLDER, "model_info.json")) as f:
        info = json.load(f)
    background = pd.read_csv(os.path.join(MODELS_FOLDER, "background.csv"))
    explainer = make_shap_explainer(model, background, size=50)
    return model, info, explainer


@st.cache_data
def load_data():
    return prepare_data(show_info=False)


def show_image(name):
    path = os.path.join(FIGURES_FOLDER, name)
    if os.path.exists(path):
        st.image(path)


model, info, explainer = load_everything()
train, test, inference, thresholds, district_list = load_data()

st.sidebar.header("Model")
st.sidebar.write("Best model:", info["best_model"])
st.sidebar.write(f"Test PR-AUC: {info['test_metrics']['PR-AUC']:.3f}")
st.sidebar.write(f"Test recall: {info['test_metrics']['Recall']:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Predict", "Data", "Model comparison", "SHAP and LIME",
                                        "Left out week"])

with tab1:
    col1, col2 = st.columns(2)
    district = col1.selectbox("District", district_list, index=district_list.index("Colombo"))
    week_start = col2.date_input("Week start date", value=pd.to_datetime("2026-05-04"))

    st.caption("You can leave the older weeks empty, the model fills them with the median.")
    c1, c2, c3, c4 = st.columns(4)
    cases_now = c1.number_input("Cases this week", min_value=0, value=None, placeholder="e.g. 328")
    cases_1 = c2.number_input("1 week ago", min_value=0, value=None, placeholder="e.g. 300")
    cases_2 = c3.number_input("2 weeks ago", min_value=0, value=None, placeholder="e.g. 279")
    cases_3 = c4.number_input("3 weeks ago", min_value=0, value=None, placeholder="e.g. 240")

    if st.button("Predict", type="primary"):
        if cases_now is None:
            st.warning("Please enter the cases for this week")
        else:
            row = make_one_row(district, week_start, cases_now, cases_1, cases_2, cases_3,
                               district_list, info["thresholds"])
            probability = model.predict_proba(row)[0, 1]

            if probability > 0.5:
                st.error(f"### {district}: HIGH RISK next week ({probability:.0%})")
            else:
                st.success(f"### {district}: Normal next week ({probability:.0%} chance of high risk)")
            st.write(f"High risk means more than **{thresholds[district]:.0f}** cases next week "
                     f"(75th percentile of {district}, 2007-2024).")

            st.subheader("Why? (SHAP)")
            values = get_shap_values(explainer, row)
            table = shap_table(values[0], row, district_list)
            st.write(f"An average week starts at {values.base_values[0]:.0%}. "
                     "Positive bars push towards high risk, negative bars towards normal.")
            st.bar_chart(table.set_index("feature")["shap"], horizontal=True)
            st.dataframe(table, hide_index=True)
            st.caption("SHAP explains what the model did, it does not show the causes of dengue.")

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows (2007-2024)", f"{len(train):,}")
    c2.metric("Test rows (2025-2026)", f"{len(test):,}")
    c3.metric("Districts", len(district_list))
    chosen = st.selectbox("Show weekly cases for", district_list, index=district_list.index("Colombo"),
                          key="data_tab")
    all_rows = pd.concat([train, test])
    st.line_chart(all_rows[all_rows["district"] == chosen].set_index("start_date")["cases"])
    for name in ["eda_national_trend.png", "eda_seasonality.png", "eda_distribution.png",
                 "eda_autocorrelation.png", "eda_class_balance.png"]:
        show_image(name)

with tab3:
    st.subheader("Validation 2024 (used to choose the model)")
    st.dataframe(pd.read_csv(os.path.join(REPORTS_FOLDER, "validation_metrics.csv"), index_col="Model"))
    st.subheader("Test 2025-2026")
    st.dataframe(pd.read_csv(os.path.join(REPORTS_FOLDER, "metrics.csv"), index_col="Model"))
    show_image("model_comparison.png")
    show_image("pr_curves.png")
    show_image("confusion_matrices.png")

with tab4:
    show_image("xai_shap_beeswarm.png")
    show_image("xai_shap_waterfall_colombo.png")
    show_image("xai_lime_colombo.png")
    show_image("xai_lime_stability.png")
    show_image("xai_permutation_importance.png")
    cases_file = os.path.join(XAI_FOLDER, "lime_vs_shap_cases.csv")
    if os.path.exists(cases_file):
        st.subheader("SHAP vs LIME on 10 test weeks")
        st.dataframe(pd.read_csv(cases_file), hide_index=True)

with tab5:
    st.write(f"The week of **{inference['start_date'].iloc[0].date()}** was never used for training or testing.")
    table = inference[["district", "cases_this_week", "cases_next_week", "risk_threshold"]].copy()
    table["actual"] = inference[TARGET].replace({1: "High risk", 0: "Normal"})
    table["probability"] = model.predict_proba(inference[FEATURES])[:, 1].round(3)
    table["predicted"] = ["High risk" if p == 1 else "Normal" for p in model.predict(inference[FEATURES])]
    st.metric("Correct districts", f"{(table['actual'] == table['predicted']).sum()} / {len(table)}")
    st.dataframe(table, hide_index=True)
