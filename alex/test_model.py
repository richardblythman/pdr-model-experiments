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
order_fee = 0 # fee, in percentage IE: 0.1 means 0.1% fee
models = [
    OceanModel(exchange_id,pair,timeframe),
    RichardModel1(exchange_id,pair,timeframe)
]
tp = 0.05 # TakeProfit, in percentage (IE: 0.1%). Ignored if 0
sl = 0.05 # StopLoss, in percentage.  Ignored if 0
ACTUAL_DO_TRADE = True  # Set this to true if you actually want to trade. THIS MAY DRAIN YOUR CEX ACCOUNT, don't whine if it does :)
## END EDIT ME    


exchange_class = getattr(ccxt, exchange_id)
exchange_ccxt = exchange_class({
    'apiKey': os.environ.get('API_KEY'),
    'secret': os.environ.get('API_SECRET'),
})


ts_now=int( time.time() )
results_csv_name='./results/'+exchange_id+"_"+models[0].pair+"_"+models[0].timeframe+"_"+str(ts_now)+".csv"
orders_csv_name='./results/'+exchange_id+"_"+models[0].pair+"_"+models[0].timeframe+"_"+str(ts_now)+"_orders.csv"





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
        writer.writerow(['model','pair','type','direction','amount','price','order_id','reason'])

def do_cex_order(pair,type,side,amount,price):
    order_id=None
    if ACTUAL_DO_TRADE:
        try:
            order_ccxt = exchange_ccxt.createOrder(symbol=pair,type=type,side=side,amount=amount,price=price)
            order_id=order_ccxt['id']
        except Exception as e:
            print(e)
    return order_id

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
        if type=='buy':
                #normal buy
                max_buy_order_size = float(order_books["asks"][0][1])/2 # never buy more than half
                actual_amount=min(order_size/last_price,max_buy_order_size)
                actual_price = float(order_books["asks"][0][0])
                actual_value = actual_price*actual_amount
                actual_fee = (actual_value*order_fee/100)
                actual_value -= actual_fee
                ccxt_direction='buy'
                ccxt_amount = actual_amount
                ccxt_price = actual_price
                order_id = do_cex_order(pair,"limit",ccxt_direction,ccxt_amount,ccxt_price)
        else:
                #normal sell
                max_sell_order_size = float(order_books["bids"][0][1])/2 # never buy more than half
                actual_price = float(order_books["bids"][0][0])
                actual_amount = min(order_size/last_price,max_sell_order_size)
                actual_value = actual_price*actual_amount
                actual_fee = (actual_value*order_fee/100)
                actual_value -= actual_fee
                ccxt_direction='sell'
                ccxt_amount = actual_amount
                ccxt_price = actual_price
                order_id = do_cex_order(pair,"limit",ccxt_direction,ccxt_amount,ccxt_price)
                
    else:
        #closing orders, the amounts are different
        if type=='buy':
            #this is a close sell position
            remaining = amount
            actual_value = 0
            bprice = 0
            for book in order_books["asks"]:
                size = min(book[1],remaining)
                actual_amount+=size
                actual_value+=book[0]*size
                bprice = book[0]
                remaining-=size
                if remaining<=0:
                    break
            fee= (actual_value*order_fee/100)
            actual_value-=fee
            actual_price = actual_value/actual_amount
            ccxt_direction='buy'
            ccxt_amount = actual_amount
            ccxt_price = bprice
            order_id = do_cex_order(pair,"limit",ccxt_direction,ccxt_amount,ccxt_price)
        else:
            #this is closing a buy position
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
            order_id = do_cex_order(pair,"limit",ccxt_direction,ccxt_amount,ccxt_price)
    #print(f"returning ({order_id},{actual_value},{actual_price},{actual_amount},{actual_fee})")
    #log orders
    with open(orders_csv_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow([model_name,pair,ccxt_type,ccxt_direction,ccxt_amount,ccxt_price,order_id,reason])
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
    #get spreads
    order_books=exchange_ccxt.fetchOrderBook(pair,3)    
    #check TP and SL
    for model in models:
        if current_positions[model.model_name] is not None:
            if current_positions[model.model_name]["direction"]==0: #it was a sell position
                #calculate current profit, for display purpose
                current_positions[model.model_name]["profit"]=current_positions[model.model_name]["spent"]-current_positions[model.model_name]["amount"]*last_price
                if current_positions[model.model_name]["tp"]>0 and last_price<=current_positions[model.model_name]["tp"]>0:
                    #take profit, let's close it..
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"buy",current_positions[model.model_name]["amount"],last_price,order_books,True,'tp')
                    profit = current_positions[model.model_name]["spent"]-actual_value
                    total_profit[model.model_name]+=profit
                    print(f"TP {model.model_name}: Bought back {current_positions[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    current_positions[model.model_name]=None
                    main_pd.loc[timestamp,[model.model_name+"_p"]]=profit
                    main_pd.loc[timestamp,[model.model_name+"_tp"]]=total_profit[model.model_name]
                elif current_positions[model.model_name]["sl"]>0 and last_price>=current_positions[model.model_name]["sl"]>0:
                    #stop loss, let's close it..
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"buy",current_positions[model.model_name]["amount"],last_price,order_books,True,'sl')
                    profit = current_positions[model.model_name]["spent"]-actual_value
                    total_profit[model.model_name]+=profit
                    print(f"SL {model.model_name}: Bought back {current_positions[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    current_positions[model.model_name]=None
                    main_pd.loc[timestamp,[model.model_name+"_p"]]=profit
                    main_pd.loc[timestamp,[model.model_name+"_tp"]]=total_profit[model.model_name]
            else: #it was a buy order, so we sell
                #calculate current profit, for display purpose
                current_positions[model.model_name]["profit"]=current_positions[model.model_name]["amount"]*last_price-current_positions[model.model_name]["spent"]
                if current_positions[model.model_name]["tp"]>0 and last_price>=current_positions[model.model_name]["tp"]>0:
                    #take profit, let's close it..
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"sell",current_positions[model.model_name]["amount"],last_price,order_books,True,'tp')
                    profit = actual_value - current_positions[model.model_name]["spent"]
                    total_profit[model.model_name]+=profit
                    print(f"TP {model.model_name}: Sold {current_positions[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    current_positions[model.model_name]=None
                    main_pd.loc[timestamp,[model.model_name+"_p"]]=profit
                    main_pd.loc[timestamp,[model.model_name+"_tp"]]=total_profit[model.model_name]
                elif current_positions[model.model_name]["sl"]>0 and last_price<=current_positions[model.model_name]["sl"]>0:
                    #stop loss, let's close it..
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"sell",current_positions[model.model_name]["amount"],last_price,order_books,True,'sl')
                    profit = actual_value - current_positions[model.model_name]["spent"]
                    total_profit[model.model_name]+=profit
                    print(f"SL {model.model_name}: Sold {current_positions[model.model_name]['amount']} at {last_price}, profit: {profit}")
                    main_pd.loc[timestamp,[model.model_name+"_p"]]=profit
                    main_pd.loc[timestamp,[model.model_name+"_tp"]]=total_profit[model.model_name]
                    current_positions[model.model_name]=None


    #we have a new candle
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
                    # if it was sell, we need to buy back
                    if current_positions[model.model_name]["direction"]==0:
                        order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"buy",current_positions[model.model_name]["amount"],last_price,order_books,True,'candle_close')
                        profit = current_positions[model.model_name]["spent"]-actual_value
                        total_profit[model.model_name]+=profit
                        print(f"Closing {model.model_name}: Bought back {current_positions[model.model_name]['amount']} at {last_price}, profit: {profit}")
                        current_positions[model.model_name]=None
                    else:
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
    order_books=exchange_ccxt.fetchOrderBook(pair,3)
    # predict & open order, if we don't have it already
    for model in models:
        index = main_pd.index.values[-1]
        current_prediction = main_pd.iloc[-1][model.model_name]
        if np.isnan(current_prediction):
            prediction = model.predict(main_pd.drop(columns_models, axis=1))
            main_pd.loc[index,[model.model_name]]=float(prediction)
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
                    if tp>0:
                        current_positions[model.model_name]["tp"]=current_positions[model.model_name]["price"]+(current_positions[model.model_name]["price"]*tp/100)
                    if sl>0:
                        current_positions[model.model_name]["sl"]=current_positions[model.model_name]["price"]-(current_positions[model.model_name]["price"]*sl/100)
                    print(f"New buy order on {model.model_name}: {current_positions[model.model_name]}")
                else:
                    # we sell
                    order_id, actual_value, actual_price, actual_amount, actual_fee = do_order(model.model_name,pair,"sell",None,last_price,order_books,False,'new_candle')
                    current_positions[model.model_name]= {
                        "direction": 0,
                        "price": actual_price, #we buy or sell at last price, need fancy bid/ask
                        "amount": actual_amount,
                        "spent": actual_value,
                        "tp": 0,
                        "sl": 0,
                        "profit":0
                    }
                    if tp>0:
                        current_positions[model.model_name]["tp"]=current_positions[model.model_name]["price"]-(current_positions[model.model_name]["price"]*tp/100)
                    if sl>0:
                        current_positions[model.model_name]["sl"]=current_positions[model.model_name]["price"]+(current_positions[model.model_name]["price"]*sl/100)
                    print(f"New sell order on {model.model_name}: {current_positions[model.model_name]}")
    
    print(f"\n\n\n********* Start: {datetime.fromtimestamp(ts_now)}, Order size: {order_size} {quote_token}. Candles closed so far: {total_candles-1} , results: {results_csv_name} *********")
    
    #exclude some columns from display
    print(main_pd.loc[:, ~main_pd.columns.isin(['volume','open','high','low'])].tail(15))
    print("\n Current positions:\n")
    for model in models:
        print(f"{model.model_name} -> {current_positions[model.model_name]}")
    time.sleep(20)




