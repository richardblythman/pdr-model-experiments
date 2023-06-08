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
import lightgbm as lgb
import seaborn as sns


class RichardModel1:
    def __init__(self,exchange,pair,timeframe):
        self.model_name=self.__class__.__name__
        self.model = [None] * 5
        self.n_fold = 5
        self.exchange = exchange.lower()
        self.pair = pair.replace("/", "-").lower()
        self.timeframe = timeframe
    
    def add_ta(self,dataframe):
        df = copy.deepcopy(dataframe)
        df.index.name = 'datetime'
        df['timestamp'] = pd.to_datetime(df.index.values,utc=True)
        for i, row in df.iterrows():
            df.loc[i,"timestamp"]=df['timestamp'][i].timestamp()
        df = ta.add_all_ta_features(
            df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True,
        )
        df = self.get_time_features(df)
        df = df.drop(["timestamp"], axis=1)
        return df

    def get_time_features(self,df):
        df = df.assign(sin_month=np.zeros(len(df)), cos_month=np.zeros(len(df)), sin_day=np.zeros(len(df)), cos_day=np.zeros(len(df)), sin_hour=np.zeros(len(df)), cos_hour=np.zeros(len(df)), sin_minute=np.zeros(len(df)), cos_minute=np.zeros(len(df)),)
        time_features = np.zeros((len(df),8))
        for i in range(len(time_features)):
            datetime = pd.to_datetime(df.index[i],utc=True)
            time_features[i,0] = (np.sin(2 * np.pi * datetime.month/12))
            time_features[i,1] = (np.cos(2 * np.pi * datetime.month/12))
            time_features[i,2] = (np.sin(2 * np.pi * datetime.day/31))
            time_features[i,3] = (np.cos(2 * np.pi * datetime.day/31))
            time_features[i,4] = (np.sin(2 * np.pi * datetime.hour/24))
            time_features[i,5] = (np.cos(2 * np.pi * datetime.hour/24))
            time_features[i,6] = (np.sin(2 * np.pi * datetime.minute/60))
            time_features[i,7] = (np.cos(2 * np.pi * datetime.minute/60))

        df[['sin_month','cos_minute', 'sin_day', 'cos_day' , 'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute']] = time_features
        return df
    
    
    def train_from_csv(self,path):
        data = pd.read_csv(path)
        data = data.set_index("timestamp")
        self.train_from_dataframe(data)
    
    def train_from_dataframe(self,dataframe):
        data = self.add_ta(dataframe)
        data["return"] = data["close"].diff().shift(-1) / data["close"]
        data["y"] = np.sign(data["close"].diff().shift(-1))
        data = data.dropna()
        print(data.tail(5))
        
        x = data.drop(["y", "return"], axis=1)
        r = data["return"]
        y = data["y"]
        y[y <= 0] = 0

        features = list(x.columns )
        kf = KFold(n_splits=self.n_fold)
        pred_returns = np.zeros((self.n_fold,))
        scores = np.zeros((self.n_fold,))
        importances = []
        params = {
            'early_stopping_rounds': 50,
            'objective': 'multiclass',
            'num_class': 2,
        }

        for split, (ind_train, ind_test) in enumerate(kf.split(x)):
            purge = 1
            ind_test = ind_test[purge:-purge]
            train_dataset = lgb.Dataset(x.iloc[ind_train, :],
                                y.iloc[ind_train].values, 
                                feature_name = features, 
                               )
            val_dataset = lgb.Dataset(x.iloc[ind_test, :], 
                              y.iloc[ind_test].values, 
                              feature_name = features, 
                             )
            self.model[split] = lgb.train(params = params,
                      train_set = train_dataset, 
                      valid_sets=[train_dataset, val_dataset],
                      valid_names=['tr', 'vl'],
                      num_boost_round = 5000,
                      verbose_eval = 100,     
                     )

        
    
    def predict(self,last_candles):
        main_pd = self.add_ta(last_candles)

        pred_list = np.zeros((self.n_fold,))
        conf_list = np.zeros((self.n_fold,))
        for split in range(self.n_fold):
            predict = self.model[split].predict(main_pd.values[[-1], :]) 
            pred_list[split] = predict.argmax(axis=1)[0]
            conf_list[split] = predict[:,1][0] 
        pred = np.median(pred_list, axis=0)    
        conf = np.median(conf_list, axis=0)    

        return pred   

    def pickle_model(self,path):
        for split in range(self.n_fold):
            model_name = path+"/"+self.model_name+"_"+self.exchange+"_"+self.pair+"_"+self.timeframe+"_fold"+str(split)+".pkl"
            with open(model_name, "wb") as f:
                pickle.dump(self.model[split], f)
    
    def unpickle_model(self,path):
        for split in range(self.n_fold):
            model_name = path+"/"+self.model_name+"_"+self.exchange+"_"+self.pair+"_"+self.timeframe+"_fold"+str(split)+".pkl"
            self.model[split] = pickle.load(open(model_name, "rb"))










