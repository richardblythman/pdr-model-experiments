# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import ta
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import copy
import pickle
import time
import warnings
from binance.spot import Spot
import os
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

model_name = "btc_tusd_hgbc"
with open(model_name + ".pkl", "rb") as f:
    model = pickle.load(f)

columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "base_asset",
    "quote_asset",
    "none",
]


def add_ta(asset_data):
    asset_data = ta.add_all_ta_features(
        asset_data,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )
    return asset_data


def get_order_book(book):
    bids = np.array(book["bids"])
    asks = np.array(book["asks"])
    minrows = min(bids.shape[0], asks.shape[0])
    book = pd.DataFrame(
        np.concatenate((bids[0:minrows, :], asks[0:minrows, :]), axis=1),
        columns=["bid_price", "bid_amount", "ask_price", "ask_amount"],
    )
    book = book.select_dtypes(include=["object", "int"]).astype(float)
    book["cum_bid_amt"] = np.cumsum(book["bid_amount"].values)
    book["cum_ask_amt"] = np.cumsum(book["ask_amount"].values)
    book["cum_bid_amt_usd"] = np.cumsum(book["bid_amount"] * book["bid_price"])
    book["cum_ask_amt_usd"] = np.cumsum(book["ask_amount"] * book["ask_price"])
    book["avg_bid_price"] = book["cum_bid_amt_usd"] / book["cum_bid_amt"]
    book["avg_ask_price"] = book["cum_ask_amt_usd"] / book["cum_ask_amt"]
    return book


def buy(symbol, client, amount_usd):
    flag = False
    while flag == False:
        book = get_order_book(client.depth(symbol))
        tolerance = 0.0001
        max_price_ = book["ask_price"].values[0] * (1 + tolerance)
        max_values = book[book["ask_price"] <= max_price_].iloc[0, :]
        max_amount_usd = min(amount_usd, max_values["cum_ask_amt_usd"])

        values = book[book["cum_ask_amt_usd"] >= max_amount_usd].iloc[0, :]
        quantity = round(max_amount_usd / values["avg_ask_price"], 5)
        limit_price = values["ask_price"]

        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "FOK",
            "quantity": quantity,
            "price": limit_price,
        }
        response = client.new_order(**params)
        if response["status"] == "FILLED":
            quantity = float(response["executedQty"])
            quote = float(response["cummulativeQuoteQty"])
            flag = True
        else:
            time.sleep(1)

    return quantity, quote


def sell(symbol, client, quantity):
    flag = False
    while flag == False:
        book = get_order_book(client.depth(symbol))
        values = book[book["cum_bid_amt"] > quantity].iloc[0, :]
        price = values["avg_bid_price"]
        min_price = values["bid_price"]
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "LIMIT",
            "timeInForce": "FOK",
            "quantity": quantity,
            "price": min_price,
        }
        response = client.new_order(**params)
        if response["status"] == "FILLED":
            quantity = float(response["executedQty"])
            quote = float(response["cummulativeQuoteQty"])
            flag = True
        else:
            time.sleep(1)

    return quantity, quote


def summary(log):
    profit_cash = round(log["profit_cash"], 4)
    profit_percent = round(log["profit_percent"], 4)
    acc = 0
    if log["total_candles"] > 0:
        acc = log["total_correct"] / log["total_candles"]
    print(f"Global results. Profit ${profit_cash} ({profit_percent}%), Accuracy {acc}")


API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
print(API_KEY, API_SECRET)
client = Spot(
    api_key=API_KEY, api_secret=API_SECRET, base_url="https://testnet.binance.vision"
)
symbol = "BTCUSDT"
amount_usd = 1000
account = client.account()
# %%
log = {
    "yhat": 0,
    "spent": None,
    "amt_purchased": None,
    "profit_percent": 0,
    "profit_cash": 0,
    "total_candles": 0,
    "total_correct": 0,
    "isNew": True,
}

latest_timestamp = 0
i = 0
while True:
    i = i + 1
    # Get klines of BTCUSDT at 5m interval
    data = client.klines(symbol, "5m", limit=100)
    data = pd.DataFrame(data, columns=columns)
    asset_data = data.drop(["none", "close_time"], axis=1)
    asset_data = asset_data.set_index("timestamp")
    asset_data = asset_data.select_dtypes(include=["object", "int"]).astype(float)
    asset_data = add_ta(asset_data)
    r = asset_data["close"].diff() / asset_data["close"]
    asset_data = asset_data.diff()
    asset_data["r"] = r
    asset_data = asset_data.dropna()

    if asset_data.index[-1] > latest_timestamp:
        print("New candle")
        latest_timestamp = asset_data.index[-1]
        yhat = model.predict(asset_data.values[[-2], :])[0]
        print(f"Predicted: {yhat}, close_diff {asset_data.values[-2, 3]}")

        # record accuracy
        log["total_candles"] = log["total_candles"] + 1
        y = 1 if asset_data.values[-2, 3] > 0 else 0
        if y == log["yhat"]:
            log["total_correct"] = log["total_correct"] + 1

        if yhat == 1 and log["isNew"] == True:
            new_event = True
            print("Buy")
            quantity, quote = buy(symbol, client, amount_usd)
            log.update(
                {
                    "yhat": yhat,
                    "spent": quote,
                    "amt_purchased": quantity,
                    "isNew": False,
                }
            )
        if yhat == 0 and log["yhat"] == 1 and log["isNew"] == False:
            new_event = True
            print("Sell")
            quantity, quote = sell(symbol, client, log["amt_purchased"])
            log["profit_percent"] += (quote - log["spent"]) / log["spent"]
            log["profit_cash"] += quote - log["spent"]

            log.update(
                {
                    "yhat": 0,
                    "spent": None,
                    "amt_purchased": None,
                    "isNew": True,
                }
            )
        else:
            new_event = False
            # print("Do Nothing")

    if log["yhat"] == 1 and log["isNew"] == 0:
        book = get_order_book(client.depth(symbol))
        values = book[book["cum_ask_amt"] > log["amt_purchased"]].iloc[0, :]
        price = values["avg_ask_price"]
        profit_usd = price * log["amt_purchased"] - log["spent"]
        profit_percent = 100 * (profit_usd / log["spent"])
        print(f"current candle. Profit {profit_usd} ({profit_percent})")

    else:
        if new_event:
            print("No active trades")
            new_event = False

    summary(log)
    time.sleep(10)


# %%
