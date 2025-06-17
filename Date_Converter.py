import pandas as pd

def hhmm_to_seconds(hhmm):
    hours = hhmm // 100
    minutes = hhmm % 100
    return hours * 3600 + minutes * 60

# Settings
city_name = "Columbus_Ohio"
input_file = city_name + "_50_years_weather.csv"        # Replace with your filename
output_file = city_name+"_weather_with_day_of_year.csv"
date_column = "date"                # Replace with your actual column name

# Load CSV
df = pd.read_csv(input_file)

# Convert to datetime and extract day-of-year
#df[date_column] = pd.to_datetime(df[date_column])
#df["day_of_year"] = df[date_column].dt.dayofyear

time_col = 'PGTM'
df["seconds"] = hhmm_to_seconds( df[time_col] )

# Save to new CSV
df.to_csv(output_file, index=False)
print(f"✅ Saved to {output_file}")
