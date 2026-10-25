# Downloads the dengue dataset again and saves it as a CSV file.
# I only need this if the CSV file in this folder is deleted.
#
# Original source: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health
# https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report
# The tables from those reports are collected in the denguedatahub project on GitHub.
#
# Run from the project folder:  python data/download_data.py

import os
import urllib.request

import pyreadr

URL = "https://raw.githubusercontent.com/thiyangt/denguedatahub/main/data/srilanka_weekly_data.rda"
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))
RDA_FILE = os.path.join(DATA_FOLDER, "srilanka_weekly_data.rda")
CSV_FILE = os.path.join(DATA_FOLDER, "srilanka_dengue_weekly_by_district.csv")

print("Downloading...")
urllib.request.urlretrieve(URL, RDA_FILE)

# The file is in R format (.rda), so pyreadr is used to read it
result = pyreadr.read_r(RDA_FILE)
df = result["srilanka_weekly_data"]
df.to_csv(CSV_FILE, index=False)
os.remove(RDA_FILE)

print("Saved", len(df), "rows to", CSV_FILE)
