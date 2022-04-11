# Setup

## Env variables

`ALPHA_VANTAGE_KEY` - Alpha vantage api key. https://www.alphavantage.co
`CRYPTO_API_KEY` - Crypto news api key. https://cryptonews-api.com
`TICKER` - The stock ID. E.g. AAL
`PATH_RAW_DATA` - Folder to save data as JSON and CSV.

# To load data

`python3 src/utils/fetchStockData.py` - loads the data and saves it as a csv file in `PATH_RAW_DATA`
