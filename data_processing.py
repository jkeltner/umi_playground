# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Data Processing
# This notebook downloads macro data from the Federal Reserve Economic Data (FRED) database. The data is downloaded from the FRED API and saved as a CSV file. We separated this from the main Notebook so that the main notebook could be run/edited without the need to get a FRED API key. This also allows us to have the main notebook fully executable in online environments like Binder without having the API key.

# +
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fredapi import Fred
import os

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
# -

# ## UMI Data
# Here we will take the raw data from https://www.usptart.com/umi in the umi.csv file and update some
#

umi_data = pd.read_csv('data/umi.csv')
def convert_date(date_str):
    date = datetime.strptime(date_str, '%b \'%y')
    return date.strftime('%Y-%m')
umi_data['date'] = umi_data['date'].apply(convert_date) # convert dates into a more format that will sort properly
umi_data.to_csv('data/umi_data.csv', index=False, mode='w')
umi_data.describe()

# ## Macro Data

# +
# make a copy of the UMI data and start pulling in all of our fun data sources!
macro_data = umi_data.copy()
macro_data.set_index('date', inplace=True)

# Pull in a bunch of data from the FRED API to build out a UMI + macro data set
# To see what each series is, check out https://fred.stlouisfed.org/ or
# use fred.get_series_info() -- e.g. fred.get_series_info('PSAVERT') or fred.get_series_info('UNRATE')
# You can also use fred.search() to discover new series -- e.g. fred.search('unemployment') or fred.search('savings')

fred = Fred(api_key=FRED_API_KEY)
#fred_series = ['PSAVERT', 'CORESTICKM159SFRBATL', 'MEDCPIM158SFRBCLE', 'UNRATE', 'FEDFUNDS', 'T10Y2YM']
fred_series = {
    'PSAVERT' : 'personal_savings_rate',
    'CORESTICKM159SFRBATL' : 'core_cpi',
    'MEDCPIM158SFRBCLE' : 'cpi',
    'UNRATE' : 'unemployment_rate',
    'FEDFUNDS' : 'fed_funds_rate',
    'T10Y2YM' : '10yr_2yr_treasury_spread'
}
for key,value in fred_series.items():
    data = fred.get_series(key)
    data.index = data.index.strftime('%Y-%m')
    data.name = value
    data = pd.DataFrame(data)
    macro_data = macro_data.join(data)

macro_data.to_csv('data/macro_data.csv', index=False, mode='w')
macro_data.describe()
# -


