from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import mplfinance as mpf

'''
Plot the Date, Open, High, Low, Close, and Volume

Train with: October 7th 2022 - December 29th 2023
Test on: January 1st 2024 - March 1st 2024
'''

df = pd.read_csv('SPY.csv', index_col=0, parse_dates=['Date'], usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

df_train = df.loc['2022-10-07':'2024-01-02']
df_test = df.loc['2024-01-01':'2024-03-01']

mpf.plot(df_train, type='candle', volume=True, show_nontrading=True, title='$SPY - Oct 7th 2022 to Dec 29th 2023')

'''
Prepare the data for training

Only need close prices and volumes, then
regularize them

The input data will be 59 consecutive days worth
of prices and volumes, and the output will be the
60th day's price
'''

train = np.delete(df_train.values, [0, 1, 2], axis=1)
test = np.delete(df_test.values, [0, 1, 2], axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

train_in = []
train_out = []
for i in range(len(train) - 59):
    train_in.append(train[i:i + 59])
    train_out.append(train[i + 59][0])
