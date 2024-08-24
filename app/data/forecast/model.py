import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from minio import Minio
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, RepeatedKFold
from xgboost import XGBRegressor

class ForecastModel(ABC):

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def _make_predict(self, *args, **kwargs):
        pass

    def train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float):
        split_index = int((1 - test_size) * len(X))
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        return X_train, X_test, y_train, y_test
    

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True) -> pd.DataFrame:
        n_in -= 1
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        cols.append(df.shift(0))
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        
        cols.append(df.shift(-n_out))
        names += [('var%d(t+%d)' % (j+1, n_out)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        
        if dropnan:
            agg.dropna(inplace=True)
        return agg

class ProphetModel(ForecastModel):
    def __init__(self):
        self.model = Prophet()
        self.is_trained = False
        self.eval = None

    def train(self, test_size: float, data: pd.DataFrame, target: str):
        data = data.groupby("Date")[target].sum().reset_index()
        data.rename({'Date': 'ds', target: 'y'}, axis=1, inplace=True)
        
        split_index = int((1 - test_size) * len(data))
        train_set = data.iloc[:split_index, :]
        test_set = data.iloc[split_index:, :]

        self.model.fit(train_set)
        self.is_trained = True

        forecast = self.model.predict(test_set)
        self.eval = Evaluation(test_set['ds'], forecast['yhat'], test_set['y'])
        self.model = Prophet().fit(data)

    def _make_predict(self, time_range: int) -> float:
        future = self.model.make_future_dataframe(periods=time_range)
        future = self.model.predict(future.tail(time_range))
        print(len(future[['ds', 'yhat']].to_dict('records')))
        
        return future[['ds', 'yhat']].to_dict('records')

class Evaluation:
    
    def __init__(self, ds: pd.Series, y_hat: pd.Series, y_truth: pd.Series):
        self.ds = ds
        self.y_hat = y_hat
        self.y_truth = y_truth

    def mse(self) -> float:
        return mean_squared_error(self.y_hat, self.y_truth)
    
    def mae(self) -> float:
        return mean_absolute_error(self.y_hat, self.y_truth)

    def plot(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.ds, self.y_truth, label='Real', color='blue')
        plt.plot(self.ds, self.y_hat, label='Predicted', color='orange')

        plt.title('Real vs Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_eval_df(self) -> pd.DataFrame:
        return pd.DataFrame({'ds': list(self.ds), 'y_hat': list(self.y_hat), "y_truth": list(self.y_truth)})

    def show_evaluation(self) -> None:
        print("Mean Squared Error:", self.mse())
        print("Mean Absolute Error:", self.mae())
        self.plot()
