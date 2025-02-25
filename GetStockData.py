import yfinance as yf
import pandas as pd

# Get Tesla stock data
tesla_stock = yf.Ticker("NKE")

# Fetch historical hourly data for the past 2 years
tesla_data_hourly = tesla_stock.history(period="730d", interval="1h")

# Convert the index (Datetime) to timezone-unaware
tesla_data_hourly.index = tesla_data_hourly.index.tz_localize(None)

# Save to an Excel file
hourly_filename = "Nike_stock_hourly_2_years.xlsx"
tesla_data_hourly.to_excel(hourly_filename)

print(f"Tesla hourly stock data for the past 2 years has been saved as {hourly_filename}")

