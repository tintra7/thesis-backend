import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from minio import Minio
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM, Bidirectional
# from keras.callbacks import EarlyStopping
# evaluate an xgboost regression model on the housing dataset
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_in -= 1
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]


    cols.append(df.shift(0))
    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    
    cols.append(df.shift(-n_out))
    names += [('var%d(t+%d)' % (j+1, n_out)) for j in range(n_vars)]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def train_test_split(X, y, test_size):
    split_index = int(test_size * len(X))
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

class Evaluation():
    
    def __init__(self, ds, y_hat, y_truth):
        self.ds = ds
        self.y_hat = y_hat
        self.y_truth = y_truth

    def mse(self):
        return mean_squared_error(self.y_hat, self.y_truth)
    
    def mae(self):
        return mean_absolute_error(self.y_hat, self.y_truth)

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.ds, self.y_truth, label='Real', color='blue')
        plt.plot(self.ds, self.y_hat, label='Predicted', color='orange')

        # Formatting the plot
        plt.title('Real vs Predicted Values')

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()

    def get_eval_df(self):
        return pd.DataFrame({'ds': self.ds, 'y_hat': self.y_hat, "y_truth": self.y_truth})

    
    def evaluation(self):
        print("Mean Squared Error:", self.mse())
        print("Mean Absolute Error:", self.mae())
        self.plot()

def train_with_prophet(data, test_size, target):
    # Prepare the data for Prophet
    prophet_data = data.groupby("Date")[target].sum()
    prophet_train = prophet_data.reset_index()
    prophet_train = prophet_train.rename(columns={"Date": "ds", target: "y"})
    prophet_train["ds"] = pd.to_datetime(prophet_train["ds"])
    print(target in data.columns)

    # Split the data into training and testing sets
    index = int(test_size * len(prophet_train))
    train = prophet_train.iloc[:index, :]
    test = prophet_train.iloc[index:, :]
    # Train the Prophet model

    model = Prophet()
    model.fit(train)

    # Make predictions
    forecast = model.predict(test)

    eval = Evaluation(test['ds'], forecast['yhat'], test['y'])
    eval.evaluation()
    model.fit(prophet_train)
    return model, eval


        
class ForecastModel:

    def __init__(self, data, time_range, target, test_size):
        self.xgb = XGBRegressor()
        self.data = data
        self.time_range = time_range
        self.target = target
        self.supervised_series = series_to_supervised(self.data.set_index("Date"), 30, self.time_range)
        self.X = self.supervised_series.iloc[:, :-1]
        self.y = self.supervised_series.iloc[:, -1]
        self.time_stamp = self.X.index

        index = int(0.8 * len(self.X))
        self.X_train = self.X[:index, :]
        self.y_train = self.y[:index]
        self.X_test = self.X[index:, :]
        self.y_test = self.y[index:]
        self.eval = Evaluation()



    def _forecast(self):
        if self.time_range in [7, 30, 90]:
            self.xgb.fit(self.X_train, self.y_train)

    
    
    

    
