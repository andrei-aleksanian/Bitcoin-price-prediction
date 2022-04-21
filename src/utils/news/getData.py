from datetime import datetime
import json
import requests

API_KEY = "bz5xysfb0suzuljfshu8sjokukmxdacwk52pz1ys"

# topicsOr = 'Tanalysis,Taxes,Institutions'

dataArr = []
for i in range(51, 96):
  url = f'https://cryptonews-api.com/api/v1?tickers=BTC&source=Bitcoin&items=50&page={i}&type=article&token={API_KEY}'
  r = requests.get(url)
  data = r.json()
  dataArr = dataArr + data["data"]

with open(f'data-{datetime.timestamp(datetime.now())}.json', 'w', encoding='utf-8') as f:
  json.dump(dataArr, f, ensure_ascii=False, indent=4)

print(f"Success, data length - {len(dataArr)}")
