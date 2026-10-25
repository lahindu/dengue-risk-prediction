# Step 1 and 2 - load the data, clean it, make features and split it
#
# run:  python -m src.data_prep

import numpy as np
import pandas as pd

from src.config import DATA_FILE, FEATURES, TARGET, TRAIN_END_YEAR, VALID_YEAR, RISK_PERCENTILE


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["start_date"] = pd.to_datetime(df["start.date"], format="mixed")
    df["year"] = df["year"].astype(int)   # saved as 2025.0 in the file
    df["week"] = df["week"].astype(int)
    df = df[["start_date", "year", "week", "district", "cases"]]
    df = df.sort_values(["district", "start_date"]).reset_index(drop=True)
    return df


def check_data(df):
    print("Nulls in each column:")
    print(df.isnull().sum())
    print("Duplicate rows (same district and week):", df.duplicated(["district", "start_date"]).sum())

    # each district should have a row for every week
    total_weeks = df["start_date"].nunique()
    for district, count in df.groupby("district").size().items():
        if count < total_weeks:
            print(f"{district} is missing {total_weeks - count} week(s)")

    print("Weekly cases -> median:", df["cases"].median(), " mean:", round(df["cases"].mean(), 1),
          " max:", df["cases"].max())


def fill_missing_weeks(df, show_info=True):
    # Puttalam is missing one week. I add the missing row and fill the cases
    # with the average of the week before and after (linear interpolation)
    all_weeks = df[["start_date", "year", "week"]].drop_duplicates("start_date")

    full = []
    for district in df["district"].unique():
        weeks = all_weeks.copy()
        weeks["district"] = district
        full.append(weeks)
    full = pd.concat(full)

    df = full.merge(df[["start_date", "district", "cases"]], on=["start_date", "district"], how="left")
    df = df.sort_values(["district", "start_date"]).reset_index(drop=True)
    if show_info:
        print("Missing weeks found:", df["cases"].isnull().sum())

    df["cases"] = df.groupby("district")["cases"].transform(lambda x: x.interpolate())
    df["cases"] = df["cases"].round()
    return df


def make_features(df, district_list):
    # IMPORTANT: shift inside each district, otherwise the last week of one district
    # becomes the "last week" of the next district
    grouped = df.groupby("district")["cases"]

    df["cases_this_week"] = df["cases"]
    df["cases_1_week_ago"] = grouped.shift(1)
    df["cases_2_weeks_ago"] = grouped.shift(2)
    df["cases_3_weeks_ago"] = grouped.shift(3)
    df["average_last_4_weeks"] = (df["cases_this_week"] + df["cases_1_week_ago"]
                                  + df["cases_2_weeks_ago"] + df["cases_3_weeks_ago"]) / 4
    df["change_from_last_week"] = df["cases_this_week"] - df["cases_1_week_ago"]
    df["month"] = df["start_date"].dt.month

    # district name -> number (Ampara = 0, Anuradhapura = 1, ...)
    df["district_code"] = df["district"].apply(lambda name: district_list.index(name))

    # this is what we want to predict, so it is NOT used as a feature
    df["cases_next_week"] = grouped.shift(-1)
    return df


def make_target(df):
    # Colombo normally has way more cases than Mannar, so each district gets
    # its own threshold. Only the training years are used to calculate it.
    train_years = df[df["year"] <= TRAIN_END_YEAR]
    thresholds = train_years.groupby("district")["cases"].quantile(RISK_PERCENTILE)

    df["risk_threshold"] = df["district"].map(thresholds)
    df[TARGET] = (df["cases_next_week"] > df["risk_threshold"]).astype(int)

    df = add_threshold_columns(df)
    return df, thresholds


def add_threshold_columns(df):
    # how far above/below the district's normal level (log so it works for big and small districts)
    # 0 = exactly at the threshold, positive = above it
    df["cases_vs_threshold"] = np.log1p(df["cases_this_week"]) - np.log1p(df["risk_threshold"])
    df["avg4_vs_threshold"] = np.log1p(df["average_last_4_weeks"]) - np.log1p(df["risk_threshold"])
    return df


def split_data(df):
    # first 3 weeks of each district have no lag values, last week has no next week
    df = df.dropna(subset=["cases_3_weeks_ago", "cases_next_week"])

    # keep the newest week out completely, I use it later to test inference on new data
    last_week = df["start_date"].max()
    inference = df[df["start_date"] == last_week]
    df = df[df["start_date"] < last_week]

    train = df[df["year"] <= TRAIN_END_YEAR]
    test = df[df["year"] > TRAIN_END_YEAR]
    return train, test, inference


def split_train_valid(train):
    # 2007-2023 to fit the models, 2024 to compare them
    fit_part = train[train["year"] < VALID_YEAR]
    valid_part = train[train["year"] == VALID_YEAR]
    return fit_part, valid_part


def prepare_data(show_info=True):
    df = load_data()
    if show_info:
        check_data(df)
    df = fill_missing_weeks(df, show_info)
    district_list = sorted(df["district"].unique())
    df = make_features(df, district_list)
    df, thresholds = make_target(df)
    train, test, inference = split_data(df)

    if show_info:
        print("Train rows:", len(train), f"({train['year'].min()}-{train['year'].max()})")
        print("Test rows:", len(test), f"({test['year'].min()}-{test['year'].max()})")
        print("Held out week:", inference["start_date"].iloc[0].date(), "-", len(inference), "districts")
        print(f"High risk weeks -> train {train[TARGET].mean():.1%}, test {test[TARGET].mean():.1%}")

    return train, test, inference, thresholds, district_list


def make_one_row(district, week_start, cases_now, cases_1, cases_2, cases_3, district_list, thresholds):
    # builds one row from the values the user types in the app.
    # empty boxes come in as None -> NaN, and the model pipeline fills them with the median
    values = []
    for v in [cases_now, cases_1, cases_2, cases_3]:
        if v is None:
            values.append(np.nan)
        else:
            values.append(float(v))

    if np.isnan(values).all():
        average = np.nan
    else:
        average = np.nanmean(values)

    week_start = pd.to_datetime(week_start)
    row = pd.DataFrame([{
        "cases_this_week": values[0],
        "cases_1_week_ago": values[1],
        "cases_2_weeks_ago": values[2],
        "cases_3_weeks_ago": values[3],
        "average_last_4_weeks": average,
        "change_from_last_week": values[0] - values[1],
        "week": min(week_start.isocalendar().week, 52),
        "month": week_start.month,
        "district_code": district_list.index(district),
        "risk_threshold": float(thresholds[district]),
    }])
    row = add_threshold_columns(row)
    return row[FEATURES]


if __name__ == "__main__":
    train, test, inference, thresholds, district_list = prepare_data()
    print(train[["start_date", "district"] + FEATURES + [TARGET]].head())
