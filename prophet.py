## Forecasting Facebook Stock Prices using Prophet

# Check https://facebook.github.io/prophet/docs/quick_start.html#python-api

import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from fbprophet import Prophet

## Define the time frame for this project


start = datetime.datetime(2012,5, 18)
end = datetime.datetime(2020, 1, 23)

## Import Data from Yahoo Finance

df = web.DataReader(["FB"], 'yahoo', start, end)
df.tail()

AdjClose = df['Adj Close']
AdjClose.tail()

# Plot the Prices

AdjClose.plot(label='FB')

# Daily Returns and cumulative returns

daily_returns = AdjClose.pct_change()

daily_returns.head()

daily_returns.plot(label='Facebook Daily Returns', grid=False)

cum_returns = (daily_returns + 1).cumprod()

cum_returns.plot(label='Facebook Cumulative Returns')

df.set_index('Date', inplace=True, drop=False)

# Predict

df.info
df.describe

stock_price = df[['Date','Adj Close']]

# Readings - https://github.com/NGYB/Stocks/blob/master/StockPricePrediction_fh21/StockPricePrediction_v2_prophet.ipynb

