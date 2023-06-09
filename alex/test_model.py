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
from datetime import datetime

from models.OceanModel import OceanModel
from models.richard1 import RichardModel1
warnings.filterwarnings("ignore")

## EDIT ME
exchange_id = 'binance'
pair='BTC/TUSD'
timeframe='5m'
quote_token='TUSD'
order_size=100  # order size, in quote token. IE: for BTC/TUSD, it's 100 TUSD
order_fee = 0.1 # fee, in percentage
models = [
    OceanModel(exchange_id,pair,timeframe),
    RichardModel1(exchange_id,pair,timeframe)
]
## END EDIT ME    


exchange_class = getattr(ccxt, exchange_id)
exchange_ccxt = exchange_class({
    'apiKey': os.environ.get('API_KEY'),
    'secret': os.environ.get('API_SECRET'),
})


ts_now=int( time.time() )
results_csv_name='./results/'+exchange_id+"_"+models[0].pair+"_"+models[0].timeframe+"_"+str(ts_now)+".csv"

columns_short = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]
hits={}
total_profit={}
current_orders={}
total_candles=0
columns_models = ['diff']  # percentage difference vs previous candle
for model in models:
    model.unpickle_model("./trained_models")
    columns_models.append(model.model_name) # prediction column.  0 or 1
    columns_models.append(model.model_name+"_m") # match column. IE: if prediction was right or not
    columns_models.append(model.model_name+"_h") # hits   IE:  model accuracy over time
    hits[model.model_name] = 0
    columns_models.append(model.model_name+"_p") # profit per candle
    columns_models.append(model.model_name+"_tp") # total profits
    total_profit[model.model_name] = 0
    current_orders[model.model_name]=None

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
        writer.writerow(all_columns)

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
    timestamp = main_pd.index.values[-2]
    last_price = main_pd.iloc[-1]['close']
    if last_finalized_timestamp<timestamp:
        total_candles+=1
        last_finalized_timestamp = timestamp
        main_pd.loc[timestamp,["diff"]]=(main_pd.iloc[-2]['close']*100/main_pd.iloc[-3]['close'])-100
        should_write = False
        for model in models:
            prediction = main_pd.iloc[-2][model.model_name]
            if not np.isnan(prediction):
                should_write = True
                match = False
                if float(prediction)>0 and main_pd.iloc[-2]['close']>main_pd.iloc[-3]['close']:
                    match=True
                elif float(prediction)<1 and main_pd.iloc[-2]['close']<main_pd.iloc[-3]['close']:
                    match=True
                main_pd.loc[timestamp,[model.model_name+"_m"]]=match
                if match:
                    hits[model.model_name]+=1
                # update hits
                main_pd.loc[timestamp,[model.model_name+"_h"]]=round(hits[model.model_name]/(total_candles-1),4)*100
                # if we have an order, close it
                if current_orders[model.model_name]:
                    # if it was sell, we need to buy back
                    if current_orders[model.model_name]["direction"]==0:
                        income = last_price*current_orders[model.model_name]["amount"]
                        income -= (income*order_fee/100)
                        profit = current_orders[model.model_name]["spent"]-income
                        total_profit[model.model_name]+=profit
                        print(f"Closing {model.model_name}: Bought back {current_orders[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    else: #if it was a buy order, we sell
                        income = last_price*current_orders[model.model_name]["amount"]
                        income -= (income*order_fee/100)
                        profit = income-current_orders[model.model_name]["spent"]
                        total_profit[model.model_name]+=profit
                        print(f"Closing {model.model_name}: Sold {current_orders[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    current_orders[model.model_name]=None
                    main_pd.loc[timestamp,[model.model_name+"_p"]]=profit
                main_pd.loc[timestamp,[model.model_name+"_tp"]]=total_profit[model.model_name]
        if should_write:
            with open(results_csv_name, 'a') as f:
                writer = csv.writer(f)
                row = [
                    main_pd.index.values[-2],
                    main_pd.iloc[-2]['open'],
                    main_pd.iloc[-2]["high"],
                    main_pd.iloc[-2]["low"],
                    main_pd.iloc[-2]["close"],
                    main_pd.iloc[-2]["volume"],
                ]
                for model in models:
                     row.append(main_pd.iloc[-2][model.model_name])
                     row.append(main_pd.iloc[-2][model.model_name+"_m"])
                     row.append(main_pd.iloc[-2][model.model_name+"_h"])
                     row.append(main_pd.iloc[-2][model.model_name+"_p"])
                     row.append(main_pd.iloc[-2][model.model_name+"_tp"])
                writer.writerow(row)
    # predict & open order, if we don't have it already
    for model in models:
        index = main_pd.index.values[-1]
        current_prediction = main_pd.iloc[-1][model.model_name]
        if np.isnan(current_prediction):
            prediction = model.predict(main_pd.drop(columns_models, axis=1))
            main_pd.loc[index,[model.model_name]]=float(prediction)
            # open order
            current_orders[model.model_name]= {
                "direction": float(prediction),
                "price": last_price, #we buy or sell at last price, need fancy bid/ask
                "amount": order_size/last_price
            }
            current_orders[model.model_name]["spent"]=current_orders[model.model_name]["price"]*current_orders[model.model_name]["amount"]
            if current_orders[model.model_name]["direction"]==0:
                print(f"New order on {model.model_name}: Sold {current_orders[model.model_name]['amount']} at {current_orders[model.model_name]['price']}, got {current_orders[model.model_name]['spent']}")
            else:
                print(f"New order on {model.model_name}: Bought {current_orders[model.model_name]['amount']} at {current_orders[model.model_name]['price']}, spent {current_orders[model.model_name]['spent']}")
    
    print(f"\n\n\n********* Start: {datetime.fromtimestamp(ts_now)}, Order size: {order_size} {quote_token}. Candles closed so far: {total_candles-1} , results: {results_csv_name} *********")
    
    #exclude some columns from display
    

    print(main_pd.loc[:, ~main_pd.columns.isin(['volume','open','high','low'])].tail(15))
    time.sleep(20)
