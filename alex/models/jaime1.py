import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from scipy import signal
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import copy
import pickle



class JaimeModel1:
    def __init__(self,exchange,pair,timeframe):
        self.model_name=self.__class__.__name__
        self.model = None
        self.exchange = exchange.lower()
        self.pair = pair.replace("/", "-").lower()
        self.timeframe = timeframe
    
    def train_from_csv(self,path):
        data = pd.read_csv(path)
        data = data.set_index("timestamp")
        self.train_from_dataframe(data)
    
    def add_ta(self,dataframe):
        dataframe = ta.add_all_ta_features(
            dataframe,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True,
        )
        return dataframe
    
    def train_from_dataframe(self,dataframe):
        data = self.add_ta(dataframe)
        data["y"] = np.sign(data["close"].diff().shift(-1))
        #data = data.drop(["none", "close_time"], axis=1)
        data = data.dropna()
        
        x = data.drop(["y"], axis=1).values
        y = data["y"].values
        y[y <= 0] = 0

        print("creating model")
        self.model = Pipeline(
            steps=[
                ("scaller", MaxAbsScaler()),
                ("Predictor", LogisticRegression(max_iter=1000)),
            ]
        )
        self.model.fit(x, y)
    
    def predict(self,last_candles):
        main_pd = copy.deepcopy(last_candles)
        main_pd = self.add_ta(main_pd)
        main_pd = main_pd.dropna()
        yhat = self.model.predict(main_pd.values[[-1], :])
        return yhat   

    def pickle_model(self,path):
        model_name=path+"/"+self.exchange+"_"+self.pair+"_"+self.timeframe+".pkl"
        with open(model_name, "wb") as f:
            pickle.dump(self.model, f)
    
    def unpickle_model(self,path):
        model_name=path+"/"+self.exchange+"_"+self.pair+"_"+self.timeframe+".pkl"
        with open(model_name, "rb") as f:
            self.model = pickle.load(f)
