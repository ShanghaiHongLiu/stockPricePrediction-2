

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas_datareader import data as web

from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


def get_stock(ticker, start_date, end_date):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Date'] = df.index
        df.index = pd.RangeIndex(len(df.index))
        col_list = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[col_list]
        return df
    except Exception as error:
        print("error")


def compute_result(ticker):

############################################################################################
    # start of LSTM model

    # get data for training set
    start_date = '2014-01-01'
    end_date = '2018-12-31'
    df_train = get_stock(ticker, start_date, end_date)

    # get the df_train['Adj Close'] column
    training_set = df_train.iloc[:, 5:6].values

    # using minmaxscaler to scale the the data ranging from 0 to 1
    sc = MinMaxScaler(feature_range=(0, 1))
    # fit transform is used for initial fitting
    training_set_scaled = sc.fit_transform(training_set)

    # create x training set with time step of 60, which means 60 days of adj close price
    x_train = []
    # create y training set for storing output
    y_train = []
    for i in range(60, df_train.shape[0]):
        x_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    # turn x and y train set from list to numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data so that it can fit into Keras recurrent layer (batch_size, timesteps, input_dim)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # initializing the neural network
    model = Sequential()

    # adding 4 LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(units=1))

    # metrics=['accuracy']
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs= 100, batch_size=32)


    # get data for testing
    df_test= get_stock(ticker, '2019-01-01', '2019-01-31')
    real_adjclose = df_test.iloc[:, 5:6].values

    dataset_total = pd.concat((df_train['Open'], df_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(df_test) - 60:].values
    # reshape the inputs data to array with len(inputs)
    inputs = inputs.reshape(-1, 1)
    # use transform instead of fit transform because mean and sd is already calculated
    inputs = sc.transform(inputs)

    x_test = []
    for i in range(60, 60+df_test.shape[0]):
        x_test.append(inputs[i - 60:i, 0])


    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    predicted_adjclose_price = model.predict(x_test)
    predicted_adjclose_price = sc.inverse_transform(predicted_adjclose_price)


    mse = mean_squared_error(real_adjclose, predicted_adjclose_price)
    rmse  = sqrt(mse)
    print(ticker, ": LSTM RMSE", rmse)



#####################################################################################
    # start of linear regression model
    start_date='2014-01-01'
    end_date='2019-01-31'
    df = get_stock(ticker, start_date, end_date)
    df1 = get_stock(ticker, start_date, end_date)
    df.set_index(['Date'], inplace = True)
    df1.set_index(['Date'], inplace = True)

    forecast_col = 'Adj Close'
    forecast_out = 60
    # int(math.ceil(0.017*len(df)))
    df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100.0
    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
    df.fillna(-99999, inplace=True)
    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.0176)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    forecast_set = clf.predict(X_lately)


############################################################################################
    # plot result
    style.use('ggplot')

    one_day = 86400
    df['Forecast'] = np.nan
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    d1 = np.array(df1[1258:]['Adj Close'])
    d2 = np.array(df[1258:]['Forecast'])

    y_true = d1
    y_pred = d2
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    print(ticker, ": Linear Regression RMSE", rmse)

    x = [2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32]

    style.use('ggplot')
    plt.plot(x, real_adjclose, "x-", label=ticker + ' Actual Price')
    plt.plot(x, predicted_adjclose_price, "+-", label=ticker + ' LSTM Predicted Price')
    plt.title(ticker + ' Price Prediction Jan 2019')
    plt.xlabel('Date')
    plt.ylabel(ticker + ' Adjusted Close Price')
    plt.plot(x, d2, "+-", label=ticker + ' Linear Regression Predicted Price')
    plt.legend()
    plt.show()

def main():
    ticker_list = ["COF", "MSFT", "GM", "BAC", "IBM"]
    for ticker in ticker_list:
        compute_result(ticker)

main()



# testing how does new branch work on github
