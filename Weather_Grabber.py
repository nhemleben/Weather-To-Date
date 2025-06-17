import requests
import pandas as pd
import time

# SETTINGS
NOAA_TOKEN = "YOUR_NOAA_API_TOKEN_HERE"
STATION_ID = "GHCND:USW00012960"  # Example: Houston Intercontinental Airport
START_YEAR = 1975
END_YEAR = 2024
LOCATION_NAME = "Houston, TX"

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

# NOAA allows max 1000 results per request
def fetch_yearly_data(year):
    records = []
    offset = 1
    while True:
        params = {
            "datasetid": "GHCND",
            "datatypeid": ["TMAX", "TMIN", "PRCP", "AWND"],  # Max/Min Temp, Precip, Wind
            "stationid": STATION_ID,
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "limit": 1000,
            "offset": offset,
            "units": "metric"
        }
        headers = {"token": NOAA_TOKEN}
        response = requests.get(BASE_URL, params=params, headers=headers)

        if response.status_code != 200:
            print(f"Failed to get data for {year}, status code {response.status_code}")
            break

        data = response.json().get("results", [])
        if not data:
            break

        records.extend(data)
        offset += 1000
        time.sleep(0.5)  # To avoid rate limits

    return records

def process_data(raw_data):
    df = pd.DataFrame(raw_data)
    if df.empty:
        return df
    df = df.pivot_table(index='date', columns='datatype', values='value', aggfunc='first').reset_index()
    df.columns.name = None
    return df

# Fetch and save all years
all_data = []
for year in range(START_YEAR, END_YEAR + 1):
    print(f"Fetching data for {year}...")
    year_data = fetch_yearly_data(year)
    df = process_data(year_data)
    if not df.empty:
        all_data.append(df)

# Combine and save
if all_data:
    final_df = pd.concat(all_data)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df.sort_values('date', inplace=True)
    final_df.to_csv(f"{LOCATION_NAME.replace(',', '').replace(' ', '_')}_50_years_weather.csv", index=False)
    print("✅ Data saved successfully!")
else:
    print("❌ No data collected.")

