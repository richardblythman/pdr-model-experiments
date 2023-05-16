# %%
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

model_name = "btc_usdt"
btc_data = pd.read_csv("../BTCUSDT_1h_.csv")
btc_data = btc_data.set_index("timestamp")

print("Extracting technical indicators")
btc_data = ta.add_all_ta_features(
    btc_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)


print("Format data")
data = copy.deepcopy(btc_data)
data["y"] = np.sign(data["close"].diff().shift(-1))
data = data.drop(["none", "close_time"], axis=1)
data = data.dropna()

# %%
x = data.drop(["y"], axis=1).values
y = data["y"].values
y[y <= 0] = 0

print("creating model")
model = Pipeline(
    steps=[
        ("scaller", MaxAbsScaler()),
        ("Predictor", LogisticRegression(max_iter=1000)),
    ]
)
model.fit(x, y)

# %%
import pickle

with open(model_name + ".pkl", "wb") as f:
    pickle.dump(model, f)

# %%
