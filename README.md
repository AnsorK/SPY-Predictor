# $SPY Predictor 🕵️
Uncovering correlation within the $SPY ETF using a Long Short-Term Memory (LSTM) Neural Network in PyTorch. Knowing 60 days of consecutive $SPY close prices allows me to predict the 61st day's close price with 78.05% accuracy*

*Note: Only for 1/2/2024 - 3/1/2024 and allowing for 1% discrepancy

## Charts
Collected historical data on daily $SPY prices from https://finance.yahoo.com/quote/SPY/history. Data is from January 29th, 1993, to March 8th, 2024.

Trained the model with closing prices from October 7th, 2022, to December 29th, 2023.
![SPY Chart](SPY_chart.png)

Visual of the training process.
![MSE](mse.png)

Using the finished model to predict prices.
![Output](output.png)

## Background
A *Neural Network* is an intricate series of equations that transforms data (numbers). Data is transformed at several *activation nodes*. The data goes through several *layers* of *activation nodes* before arriving to a conclusion.

*Backwards propagation* is when the network updates the weights and biases used for linear transformation of data. This is to make the expected output and actual output as close as possible, i.e. lower the error.

The whole thing gets updated during an *epoch*, which is one whole pass through the network. Multiple *epochs* are run to lower the error.

A *Recurrent Neural Network* (*RNN*), extends a regular *Neural Network* by transforming data within a layer several times. It is used to work with temporal (time-based) data, because present values depend on the past.

The *LSTM Neural Network* is an *RNN* that uses long-lived and short-lived values during the training process. It is particularly designed to address the pitfalls of regular *RNN*'s.

## Inspiration
https://www.kaggle.com/code/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch/notebook
