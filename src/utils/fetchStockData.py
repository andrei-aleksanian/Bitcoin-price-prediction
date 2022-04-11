import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.environ.get("CRYPTO_COMPARE_KEY")
PATH_RAW_DATA = os.environ.get("PATH_RAW_DATA")
TICKER = os.environ.get("TICKER")
url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={TICKER}&tsym=USD&limit=1825&apikey={KEY}'

r = requests.get(url)
data = r.json()

with open(f'{PATH_RAW_DATA}-{TICKER}.json', 'w', encoding='utf-8') as f:
  json.dump(data, f, ensure_ascii=False, indent=4)
