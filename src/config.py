# All the settings for the project are kept here so I only have to change them in one place

import os

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_FOLDER, "data", "srilanka_dengue_weekly_by_district.csv")
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
REPORTS_FOLDER = os.path.join(PROJECT_FOLDER, "reports")
FIGURES_FOLDER = os.path.join(REPORTS_FOLDER, "figures")
XAI_FOLDER = os.path.join(REPORTS_FOLDER, "xai")

# make the output folders if they are not there
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(XAI_FOLDER, exist_ok=True)

# Time split (I don't use a random split because this is time series data)
#   2007 - 2023  training
#   2024         validation -> used to pick the best model
#   2025 - 2026  test       -> only used at the end
VALID_YEAR = 2024
TRAIN_END_YEAR = 2024

# "High risk" = next week's cases are above the 75th percentile of that district
RISK_PERCENTILE = 0.75

FEATURES = [
    "cases_this_week",
    "cases_1_week_ago",
    "cases_2_weeks_ago",
    "cases_3_weeks_ago",
    "average_last_4_weeks",
    "change_from_last_week",
    "week",
    "month",
    "district_code",
    "cases_vs_threshold",   # how far this week is above/below the district's normal level
    "avg4_vs_threshold",    # same thing for the 4 week average
]
TARGET = "high_risk"

# nicer names for the charts
FEATURE_NAMES = {
    "cases_this_week": "Cases this week",
    "cases_1_week_ago": "Cases 1 week ago",
    "cases_2_weeks_ago": "Cases 2 weeks ago",
    "cases_3_weeks_ago": "Cases 3 weeks ago",
    "average_last_4_weeks": "Average last 4 weeks",
    "change_from_last_week": "Change from last week",
    "week": "Week of year",
    "month": "Month",
    "district_code": "District",
    "cases_vs_threshold": "Cases vs threshold (log)",
    "avg4_vs_threshold": "4 week avg vs threshold (log)",
}

# columns that get log + scaling, and columns that only get scaling
COUNT_COLUMNS = ["cases_this_week", "cases_1_week_ago", "cases_2_weeks_ago",
                 "cases_3_weeks_ago", "average_last_4_weeks"]
OTHER_COLUMNS = ["change_from_last_week", "week", "month",
                 "cases_vs_threshold", "avg4_vs_threshold"]

MODEL_NAMES = [
    "Persistence rule",      # simple rule, no ML (what a health officer would do)
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Soft Voting",
    "Stacking",
]

RANDOM_STATE = 42
