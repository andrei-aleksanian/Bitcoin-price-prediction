import pandas as pd

# import os
# import datetime as dt
# import json
# from dotenv import load_dotenv
# load_dotenv(override=True)

# PATH = os.environ.get("PATH_RAW_DATA")
# TICKER = os.environ.get("TICKER")


def loadData():
  csv_path = "data/BTC-USD.csv"
  df = pd.read_csv(csv_path, parse_dates=['Date'])
  df = df.sort_values('Date')
  return df

# def loadData():
#   """Deprecated. Was used earlier for an older source"""
#   PATH_RAW_DATA = f"data/data-BTC.json"
#   PATH_RAW_DATA_CSV = "data/data-BTC.csv"
#   print(f"Looking for {TICKER}")

#   if not os.path.exists(PATH_RAW_DATA_CSV):
#     if not os.path.exists(PATH_RAW_DATA):
#       raise FileNotFoundError

#     f = open(PATH_RAW_DATA)
#     data = json.load(f)
#     data = data['Data']["Data"]
#     df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
#     for v in data:
#       date = dt.datetime.utcfromtimestamp(v["time"]).strftime('%Y-%m-%d')
#       data_row = [date, float(v['low']), float(v['high']),
#                   float(v['close']), float(v['open'])]
#       df.loc[-1, :] = data_row
#       df.index = df.index + 1
#     print('Data saved to : %s' % PATH_RAW_DATA_CSV)
#     df.to_csv(PATH_RAW_DATA_CSV)

#   # If the data is already there, just load it from the CSV
#   else:
#     print('File already exists. Loading data from CSV')
#     df = pd.read_csv(PATH_RAW_DATA_CSV)

#   return df
