import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.environ.get("ALPHA_VANTAGE_KEY")
PATH_RAW_DATA = os.environ.get("PATH_RAW_DATA")
TICKER = os.environ.get("TICKER")
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={TICKER}&apikey={KEY}&outputsize=full'
r = requests.get(url)
data = r.json()

with open(f'{PATH_RAW_DATA}-{TICKER}.json', 'w', encoding='utf-8') as f:
  json.dump(data, f, ensure_ascii=False, indent=4)

print("Success")
