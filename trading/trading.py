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

warnings.filterwarnings("ignore")

model_name = "btc_usdt"
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
# %%

from binance.spot import Spot

client = Spot()
i = 0
results = pd.DataFrame({"price": [20000], "yhat": [0]})
while True:
    i = i + 1
    # Get klines of BTCUSDT at 5m interval
    data = client.klines("BTCUSDT", "5m", limit=100)
    asset_data = pd.DataFrame(data, columns=columns)
    asset_data = asset_data.drop(["none", "close_time"], axis=1)
    asset_data = asset_data.set_index("timestamp")
    asset_data = asset_data.select_dtypes(include=["object", "int"]).astype(float)

    asset_data = ta.add_all_ta_features(
        asset_data,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )
    asset_data = asset_data.dropna()
    yhat = model.predict(asset_data.values[[-1], :])
    results = results.append(
        {"price": asset_data["close"].iloc[-1], "yhat": yhat}, ignore_index=True
    )
    tmp = copy.deepcopy(results)
    tmp["returns"] = (tmp["price"].diff().shift(-1)) / tmp["price"]
    tmp["profits"] = tmp["returns"] * tmp["yhat"]
    tmp = tmp.dropna()
    print("Total Profits: ", np.sum(tmp["profits"]))
    print("Profit latest frame: ", tmp["profits"].values[-1])
    time.sleep(60 * 5)


# %%
