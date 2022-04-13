# Setup

## Env variables

`ALPHA_VANTAGE_KEY` - Alpha vantage api key. https://www.alphavantage.co
`CRYPTO_API_KEY` - Crypto news api key. https://cryptonews-api.com
`TICKER` - The stock ID. E.g. AAL
`PATH_RAW_DATA` - Folder to save data as JSON and CSV.

## To load data

`python3 src/utils/fetchStockData.py` - loads the data and saves it as a csv file in `PATH_RAW_DATA`

# Environment

## Notebooks

`main.ipynb` - the extensive experiment
`colab.ipynb` - the colab version. Used for mainly retraining the model. Not always of interest in Git.

## Connect to colab by ssh

Open a colab notebook with cloudflare, run the command:

```
!pip install colab_ssh --upgrade
from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="mypassword")
```

And importantly run an infinite loop in a separate cell:

`while True : pass`

Connect to the colab machine by ssh. For example, using VSC Remote SSH package.

```
Command + Shift + P
Remote-SSH: Connect To Host
Paste the host name from colab notebook
```

Then install all dependencies:

`pip3 install -r requirements.txt`

Clone the repository if required

`git clone https://github.com/andrei-aleksanian/project.git`

Install VSC packages: Python and Jupyter
