import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from io import StringIO
import re
import os

# get json object
with open('src/data/news/bitcoin.json', 'r') as myfile:
  data = myfile.read()
newsList = json.loads(data)
# get article link
# go thru each article in for loop and get first n sentences

articleContent = []

count = 1

for article in newsList:
  url = article['news_url']
  req = requests.get(url)
  page = req.content
  soup = BeautifulSoup(page, 'html5lib')
  paragraphs = soup.find_all('div', class_='m-detail--body')

  text = ''
  if len(paragraphs) == 0:
    print(f"Error in row - {count}")
    text = "ERRORs"
  else:
    text = ' '.join(re.split(r'(?<=[.:;])\s', paragraphs[0].get_text())[:5])

  articleContent.append(text)
  article['paragraph'] = text
  count += 1

jsonNews = json.dumps(newsList)
df = pd.read_json(StringIO(jsonNews))
df.to_csv('src.data/news/bitcoin.csv', index=None)
