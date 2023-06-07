import sys

sys.path.append("../")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from scipy import signal
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import copy
import pickle


class OceanModel:
    def __init__(self, exchange, pair, timeframe):
        self.model_name = self.__class__.__name__
        self.model = None
        self.exchange = exchange.lower()
        self.pair = pair.replace("/", "-").lower()
        self.timeframe = timeframe
        self.predictor = LogisticRegression(max_iter=1000)

    def select_predictor(self, model_id):
        models = {
            "1": LogisticRegression(max_iter=1000),
            "2": HGBC(),
        }
        self.predictor = models[model_id]

    def feature_extraction(self, dataframe):
        dataframe = ta.add_all_ta_features(
            dataframe,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True,
        )
        dataframe = dataframe.diff()
        return dataframe

    def format_train_data(self, dataframe):
        dataframe = self.feature_extraction(dataframe)
        dataframe["labels"] = np.where(dataframe["close"].shift(-1) > 0, 1, 0)
        dataframe = dataframe.dropna()
        return dataframe

    def format_test_data(self, dataframe):
        dataframe = self.feature_extraction(dataframe)
        dataframe = dataframe.dropna()
        return dataframe

    def train_from_csv(self, path):
        data = pd.read_csv(path)
        data = data.set_index("timestamp")
        data = self.format_train_data(data)
        self.train(data)

    def train(self, dataframe):
        if self.predictor is None:
            raise Exception("Division by zero is not allowed.")

        x = dataframe.drop(["labels"], axis=1).values
        y = dataframe["labels"].values
        print("creating model")
        self.model = Pipeline(
            steps=[
                ("scaller", MaxAbsScaler()),
                ("Predictor", self.predictor),
            ]
        )
        self.model.fit(x, y)

    def predict(self, last_candles):
        dataframe = copy.deepcopy(last_candles)
        dataframe = self.feature_extraction(dataframe)
        yhat = self.model.predict(dataframe.values[[-1], :])
        return yhat

    def pickle_model(self, path):
        model_name = (
            path + "/" + self.exchange + "_" + self.pair + "_" + self.timeframe + ".pkl"
        )
        with open(model_name, "wb") as f:
            pickle.dump(self.model, f)

    def unpickle_model(self, path):
        model_name = (
            path + "/" + self.exchange + "_" + self.pair + "_" + self.timeframe + ".pkl"
        )
        with open(model_name, "rb") as f:
            self.model = pickle.load(f)
