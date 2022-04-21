import pandas as pd
df1 = pd.read_json('src/data/news/bitcoin.json')
# df2 = pd.read_json('src/data/news/bitcoin2.json')

# df = pd.concat([df1, df2], ignore_index=True, sort=False)

df1.to_csv('src/data/news/bitcoin.csv', index=None)
