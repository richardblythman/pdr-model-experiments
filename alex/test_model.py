import sys
sys.path.append('../')
import ccxt
import os
import pandas as pd
import numpy as np
import ta
import time
import pickle
import csv 
import warnings
from models.jaime1 import JaimeModel1
warnings.filterwarnings("ignore")

exchange_id = 'binance'
pair='BTC/TUSD'
timeframe='5m'


## EDIT ME
models = [
    JaimeModel1(exchange_id,pair,timeframe)
]
    


exchange_class = getattr(ccxt, exchange_id)
exchange_ccxt = exchange_class({
    'apiKey': os.environ.get('API_KEY'),
    'secret': os.environ.get('API_SECRET'),
})


ts_now=int( time.time() )
results_csv_name='./results/'+str(ts_now)+".csv"

columns_short = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

columns_models = []
for model in models:
    model.unpickle_model("./trained_models")
    columns_models.append("Prediction_"+model.model_name)
    columns_models.append("Prediction_"+model.model_name+"_match")

all_columns=columns_short+columns_models

#write csv header
size = 0
try:
    files_stats=os.stat(results_csv_name)
    size = files_stats.st_size
except:
    pass
if size==0:
     with open(results_csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(columns_short)

#read initial set of candles
candles = exchange_ccxt.fetch_ohlcv(pair, "5m")
#load past data
main_pd = pd.DataFrame(columns=all_columns)
for ohl in candles:
        ohlc= {
                'timestamp':int(ohl[0]/1000),
                'open':float(ohl[1]),
                'close':float(ohl[4]),
                'low':float(ohl[3]),
                'high':float(ohl[2]),
                'volume':float(ohl[5]),
        }
        main_pd.loc[ohlc["timestamp"]]=ohlc
main_pd = main_pd.set_index("timestamp")
last_csv_timestamp=0
last_finalized_timestamp = 0
while True:
    # Get klines 
    candles = exchange_ccxt.fetch_ohlcv(pair, timeframe="5m", limit = 3)
    #update last two candles
    for ohl in candles[-2:]:
        t = int(ohl[0]/1000)
        main_pd.loc[t,['open']]=float(ohl[1])
        main_pd.loc[t,['close']]=float(ohl[4])
        main_pd.loc[t,['low']]=float(ohl[3])
        main_pd.loc[t,['high']]=float(ohl[2])
        main_pd.loc[t,['volume']]=float(ohl[5])
    for model in models:
        prediction = model.predict(main_pd.drop(columns_models, axis=1))
        main_pd.loc[t,["Prediction_"+model.model_name]]=float(prediction)
    timestamp = main_pd.index.values[-2]
    if last_finalized_timestamp<timestamp:
         last_finalized_timestamp = timestamp
         
    print(f"\n\n\n********* TAIL-ing *********")
    print(main_pd.tail(15))
    time.sleep(20)
