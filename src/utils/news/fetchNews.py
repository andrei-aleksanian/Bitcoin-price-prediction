from datetime import datetime
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

CRYPTO_NEWS_API_KEY = os.environ.get("CRYPTO_NEWS_API_KEY")

topicsOr = 'Tanalysis,Mining,Taxes,Upgrade,Institutions'

dataArr = []
for i in range(1, 31):
  url = f'https://cryptonews-api.com/api/v1?tickers=BTC,ETH&source=Coindesk,Cointelegraph&items=50&page={i}&topicOR={topicsOr}&token={CRYPTO_NEWS_API_KEY}'
  r = requests.get(url)
  data = r.json()
  dataArr = dataArr + data["data"]

with open(f'data-{datetime.timestamp(datetime.now())}.json', 'w', encoding='utf-8') as f:
  json.dump(dataArr, f, ensure_ascii=False, indent=4)

print(f"Success, data length - {len(dataArr)}")
