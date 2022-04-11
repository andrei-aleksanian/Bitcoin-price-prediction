import pandas as pd
df = pd.read_json('src/data/data-BTC.json')
df.to_csv('src/data/data-BTC.csv', index=None)
