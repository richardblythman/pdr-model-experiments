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

btc_data = pd.read_csv("BTCUSDT_5mval_.csv")
btc_data = btc_data.set_index("timestamp")

btc_data = ta.add_all_ta_features(
    btc_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)


data = copy.deepcopy(btc_data)
data["return"] = data["close"].diff().shift(-1) / data["close"]
data["y"] = np.sign(data["close"].diff().shift(-1))
data = data.drop(["none", "close_time"], axis=1)
data = data.dropna()


x = data.drop(["y", "return"], axis=1).values
r = data["return"].values
y = data["y"].values
y[y <= 0] = 0

yhat = model.predict(x)
pred_returns = np.sum(r[yhat == 1])
opt_returns = np.sum(r[y == 1])

print(f"Optimal returns: {opt_returns}")
print(f"Predicted returns: {pred_returns}")
print(f"Accuracy : {acc(yhat,y)}")
# %%
