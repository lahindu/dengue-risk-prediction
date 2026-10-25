# Step 1 - exploratory data analysis (EDA)
# Looking at the data before choosing any model.
#
# run:  python -m src.eda    (charts are saved in reports/figures)

import os

import matplotlib
matplotlib.use("Agg")   # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_FOLDER, TARGET
from src.data_prep import (load_data, fill_missing_weeks, make_features, make_target,
                           split_data, split_train_valid)


def save_chart(file_name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, file_name), dpi=130)
    plt.close()


def main():
    raw = load_data()
    df = fill_missing_weeks(raw, show_info=False)
    district_list = sorted(df["district"].unique())
    df = make_features(df, district_list)
    df, thresholds = make_target(df)
    train, test, inference = split_data(df)
    fit_part, valid_part = split_train_valid(train)

    print("Rows:", len(raw), "| Districts:", df["district"].nunique())
    print("From", df["start_date"].min().date(), "to", df["start_date"].max().date())
    print("Total cases:", int(df["cases"].sum()))

    # 1. cases in the whole country every week
    national = df.groupby("start_date")["cases"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(national.index, national.values, color="teal", linewidth=1)
    plt.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), color="gold", alpha=0.3,
                label="Validation 2024")
    plt.axvspan(pd.Timestamp("2025-01-01"), national.index.max(), color="orange", alpha=0.2,
                label="Test 2025-2026")
    plt.title("Weekly dengue cases in Sri Lanka (all 26 districts)")
    plt.ylabel("Cases per week")
    plt.legend()
    save_chart("eda_national_trend.png")
    print("Biggest week:", national.idxmax().date(), "with", int(national.max()), "cases (2017 epidemic)")

    # 2. seasonality - which weeks of the year have more cases
    by_week = df[(df["year"] <= 2024) & (df["week"] <= 52)]   # week 53 only exists in some years
    by_week = by_week.groupby(["year", "week"])["cases"].sum().reset_index()
    by_week["percent"] = by_week["cases"] / by_week.groupby("year")["cases"].transform("sum") * 100
    season = by_week.groupby("week")["percent"].mean()
    plt.figure(figsize=(11, 4))
    plt.bar(season.index, season.values, color="teal")
    plt.title("Average % of the year's cases in each week (2007-2024)")
    plt.xlabel("Week of year (peaks around Jan and Jul = the two monsoons)")
    plt.ylabel("% of yearly cases")
    save_chart("eda_seasonality.png")

    # 3. average cases per district
    per_district = df[df["year"] <= 2024].groupby("district")["cases"].mean().sort_values()
    plt.figure(figsize=(9, 7))
    plt.barh(per_district.index, per_district.values, color="teal")
    plt.title("Average weekly cases per district (2007-2024)")
    save_chart("eda_districts.png")
    print("Colombo threshold:", thresholds["Colombo"], "| Mannar threshold:", thresholds["Mannar"])
    western = df[df["district"].isin(["Colombo", "Gampaha", "Kalutara"])]["cases"].sum()
    print(f"Western Province share of all cases: {western / df['cases'].sum():.0%}")

    # 4. the distribution is very skewed -> log transform
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df["cases"], bins=80, color="teal")
    ax[0].set_title(f"Raw weekly cases (skewness {df['cases'].skew():.1f})")
    ax[1].hist(np.log1p(df["cases"]), bins=40, color="orange")
    ax[1].set_title(f"log(1 + cases) (skewness {np.log1p(df['cases']).skew():.2f})")
    save_chart("eda_distribution.png")
    print(f"Skewness: raw {df['cases'].skew():.1f}, after log {np.log1p(df['cases']).skew():.2f}")

    # 5. autocorrelation - does this week tell us about next weeks?
    lags = list(range(1, 13))
    corr = []
    for lag in lags:
        values = []
        for district, group in df.groupby("district"):
            values.append(np.log1p(group["cases"]).autocorr(lag))
        corr.append(np.mean(values))
    plt.figure(figsize=(8, 4))
    plt.bar(lags, corr, color="teal")
    plt.ylim(0, 1)
    plt.title("Autocorrelation of log weekly cases (average of districts)")
    plt.xlabel("Lag (weeks)")
    save_chart("eda_autocorrelation.png")
    print(f"Autocorrelation lag 1: {corr[0]:.2f}, lag 4: {corr[3]:.2f}")

    # 6. class balance
    parts = {"Train 2007-2023": fit_part, "Validation 2024": valid_part, "Test 2025-2026": test}
    shares = []
    for name in parts:
        shares.append(parts[name][TARGET].mean() * 100)
    plt.figure(figsize=(7, 4))
    plt.bar(list(parts.keys()), shares, color=["teal", "gold", "orange"])
    for i in range(len(shares)):
        plt.text(i, shares[i] + 1, f"{shares[i]:.1f}%", ha="center")
    plt.ylim(0, 60)
    plt.ylabel("% high risk weeks")
    plt.title("High risk weeks in each part of the data")
    save_chart("eda_class_balance.png")
    for name, share in zip(parts.keys(), shares):
        print(f"High risk weeks in {name}: {share:.1f}%")


if __name__ == "__main__":
    main()
