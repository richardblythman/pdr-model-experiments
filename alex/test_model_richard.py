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
import datetime
import pytz

from models.OceanModel import OceanModel
from models.richard1 import RichardModel1
warnings.filterwarnings("ignore")

## EDIT ME
exchange_id = 'binance'
pair='BTC/TUSD'
timeframe='5m'
quote_token='TUSD'
order_size=1000  # order size, in quote token. IE: for BTC/TUSD, it's 100 TUSD
order_fee = 0 # fee, in percentage IE: 0.1 means 0.1% fee
models = [
    # OceanModel(exchange_id,pair,timeframe),
    RichardModel1(exchange_id,pair,timeframe)
]
tp = 0 # TakeProfit, in percentage (IE: 0.1%). Ignored if 0
sl = 0 # StopLoss, in percentage.  Ignored if 0
ACTUAL_DO_TRADE = False  # Set this to true if you actually want to trade. THIS MAY DRAIN YOUR CEX ACCOUNT, don't whine if it does :)
## END EDIT ME    


exchange_class = getattr(ccxt, exchange_id)
exchange_ccxt = exchange_class({
    'apiKey': os.environ.get('API_KEY'),
    'secret': os.environ.get('API_SECRET'),
    'timeout': 30000
})


ts_now=int( time.time() )
results_csv_name='./results/'+exchange_id+"_"+models[0].pair+"_"+models[0].timeframe+"_"+str(ts_now)+".csv"
orders_csv_name='./results/'+exchange_id+"_"+models[0].pair+"_"+models[0].timeframe+"_"+str(ts_now)+"_orders.csv"
features_csv_name='./results/'+exchange_id+"_"+models[0].pair+"_"+models[0].timeframe+"_"+str(ts_now)+"_features.csv"



columns_short = [
    "timestamp",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume"
]
hits={}
total_profit={}
current_positions={}
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
    current_positions[model.model_name]=None

all_columns=columns_short+columns_models

#write csv header for results
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

#write csv header for orders log
size = 0
try:
    files_stats=os.stat(orders_csv_name)
    size = files_stats.st_size
except:
    pass
if size==0:
     with open(orders_csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['exact time', 'expected price', 'model','pair','type','direction','amount','price','value','order_id','reason'])

def do_cex_order(pair,type,side,amount,price):
    order_id=None
    order_time=None
    if ACTUAL_DO_TRADE:
        try:
            order_ccxt = exchange_ccxt.createOrder(symbol=pair,type=type,side=side,amount=amount,price=price)
            order_id=order_ccxt['id']
            order_timestamp = order_ccxt['timestamp'] / 1000  # Divide by 1000 to convert milliseconds to seconds
            order_time = datetime.datetime.fromtimestamp(order_timestamp)
        except Exception as e:
            print(e)
    else:
        order_time = datetime.datetime.now(pytz.UTC)
    return order_id, order_time

def do_order(model_name,pair,type,amount,price,order_books,is_closing_order, reason):
    order_id = None
    actual_value = 0
    actual_price = 0
    actual_amount = 0
    actual_fee=0

    #for logging
    ccxt_type='limit'
    ccxt_direction=None
    ccxt_amount=0
    ccxt_price=0
    if not is_closing_order:
        #normal buy
        remaining_value = order_size
        for i, book in enumerate(order_books["asks"]):
            actual_price = float(book[0])
            remaining = remaining_value/actual_price
            size = min(book[1],remaining)

            current_value=actual_price*size
            actual_value+=current_value

            actual_amount+=size
            remaining_value-=current_value
            if remaining_value<=0:
                break
        actual_fee = (actual_value*order_fee/100)
        actual_value -= actual_fee
        ccxt_direction='buy'
        ccxt_amount = actual_amount
        ccxt_price = actual_price
        order_id, order_time = do_cex_order(pair,"limit",ccxt_direction,ccxt_amount,ccxt_price)
    else:
        #normal sell
        remaining = amount
        actual_value = 0
        bprice = 0
        for book in order_books["bids"]:
            size = min(book[1],remaining)
            actual_value+=book[0]*size
            bprice = book[0]
            actual_amount+=size
            remaining-=size
            if remaining<=0:
                break
        fee= (actual_value*order_fee/100)
        actual_value-=fee
        actual_price = actual_value/actual_amount
        ccxt_direction='sell'
        ccxt_amount = actual_amount
        ccxt_price = bprice
        order_id, order_time = do_cex_order(pair,"limit",ccxt_direction,ccxt_amount,ccxt_price)

    #print(f"returning ({order_id},{actual_value},{actual_price},{actual_amount},{actual_fee})")
    #log orders
    with open(orders_csv_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow([order_time, price, model_name,pair,ccxt_type,ccxt_direction,ccxt_amount,ccxt_price,actual_value,order_id,reason])
    return(order_id,actual_value,actual_price,actual_amount,actual_fee)

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
        main_pd["datetime"] = pd.to_datetime(main_pd["timestamp"], unit="s", utc=True)

#get initial set of features

for model in models:
    main_pd_features = model.add_ta(main_pd[columns_short])

#write csv header for features
size = 0
try:
    files_stats=os.stat(features_csv_name)
    size = files_stats.st_size
except:
    pass
if size==0:
     with open(features_csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','datetime','diff']+list(main_pd_features.columns)[1:])

main_pd = main_pd.set_index("timestamp")
last_csv_timestamp=0
last_finalized_timestamp = 0
while True:
    # Get klines 
    candles = exchange_ccxt.fetch_ohlcv(pair, timeframe="5m", limit = 3)

    #update last two candles
    for ohl in candles[-2:]:
        t = int(ohl[0]/1000)
        main_pd.loc[t,['datetime']]=pd.to_datetime(t, unit="s", utc=True)
        main_pd.loc[t,['open']]=float(ohl[1])
        main_pd.loc[t,['close']]=float(ohl[4])
        main_pd.loc[t,['low']]=float(ohl[3])
        main_pd.loc[t,['high']]=float(ohl[2])
        main_pd.loc[t,['volume']]=float(ohl[5])

    timestamp = main_pd.index.values[-2]
    last_price = main_pd.iloc[-2]['close']
    #get spreads
    order_books=exchange_ccxt.fetchOrderBook(pair,100) 

    # #we have a new candle
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
                if current_positions[model.model_name]:
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"sell",current_positions[model.model_name]["amount"],last_price,order_books,True,'candle_close')
                    profit = actual_value-current_positions[model.model_name]["spent"]
                    total_profit[model.model_name]+=profit
                    print(f"Closing {model.model_name}: Sold {current_positions[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    current_positions[model.model_name]=None

                    main_pd.loc[timestamp,[model.model_name+"_p"]]=profit
                    main_pd.loc[timestamp,[model.model_name+"_tp"]]=total_profit[model.model_name]
                    
                
        if should_write:
            with open(results_csv_name, 'a') as f:
                writer = csv.writer(f)
                row = [
                    main_pd.index.values[-2],
                    main_pd.iloc[-2]['datetime'],
                    main_pd.iloc[-2]['open'],
                    main_pd.iloc[-2]["high"],
                    main_pd.iloc[-2]["low"],
                    main_pd.iloc[-2]["close"],
                    main_pd.iloc[-2]["volume"],
                    main_pd.iloc[-2]["diff"],
                ]
                for model in models:
                     row.append(main_pd.iloc[-2][model.model_name])
                     row.append(main_pd.iloc[-2][model.model_name+"_m"])
                     row.append(main_pd.iloc[-2][model.model_name+"_h"])
                     row.append(main_pd.iloc[-2][model.model_name+"_p"])
                     row.append(main_pd.iloc[-2][model.model_name+"_tp"])
                writer.writerow(row)
    # get order book, in case we need it
    order_books=exchange_ccxt.fetchOrderBook(pair,100)

    # predict & open order, if we don't have it already
    for model in models:
        index = main_pd.index.values[-1]
        current_prediction = main_pd.iloc[-1][model.model_name]
        if np.isnan(current_prediction):
            prediction, main_pd_features = model.predict(main_pd.drop(columns_models  + ['datetime'], axis=1))
            main_pd.loc[index,[model.model_name]]=float(prediction)
            # df_return = pd.concat([main_pd.iloc[-2][['datetime','diff',model.model_name]], main_pd_features], axis=1)

            with open(features_csv_name, 'a') as f:
                writer = csv.writer(f)
                row = [
                    main_pd.index.values[-2],
                    main_pd.iloc[-2]['datetime'],
                    main_pd.iloc[-2]["diff"],
                ]
                row.extend(*main_pd_features)
                writer.writerow(row)

            # open order
            if current_positions[model.model_name] is None:
                if float(prediction)>0:
                    # we buy
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"buy",None,last_price,order_books,False,'new_candle')
                    current_positions[model.model_name]= {
                        "direction": 1,
                        "price": actual_price, #we buy or sell at last price, need fancy bid/ask
                        "amount": actual_amount,
                        "spent": actual_value,
                        "tp": 0,
                        "sl": 0,
                        "profit":0
                    }
                    print(f"New buy order on {model.model_name}: {current_positions[model.model_name]}")
                else:
                    pass
    
    print(f"\n\n\n********* Start: {datetime.datetime.fromtimestamp(ts_now)}, Order size: {order_size} {quote_token}. Candles closed so far: {total_candles-1} , results: {results_csv_name} *********")
    
    #exclude some columns from display
    print(main_pd.loc[:, ~main_pd.columns.isin(['volume','open','high','low'])].tail(15))
    print("\n Current positions:\n")
    for model in models:
        print(f"{model.model_name} -> {current_positions[model.model_name]}")
    time.sleep(20)




