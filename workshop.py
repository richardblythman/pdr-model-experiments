# %%
import pandas as pd
import numpy as np
import ccxt

ohlcv = ccxt.binance().fetch_ohlcv(symbol="ETH/USD", limit=1000, timeframe="5m")
# %%
data = pd.DataFrame(
    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
)

data["y"] = data["close"].shift(-1)
data = data.dropna()

y = data["y"].values
x = data.drop(["y"], axis=1).values

n = x.shape[0]
train_size = int(0.8 * n)

x_train = x[0:train_size, :]
y_train = y[0:train_size]

x_test = x[train_size:-1, :]
y_test = y[train_size:-1]
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae

model = LinearRegression()
model = model.fit(x_train, y_train)

yhat = model.predict(x_test)
score = mae(yhat, y_test)
print("MAE: ", score)

import matplotlib.pyplot as plt

plt.plot(y_test)
plt.plot(yhat)
# %% baseline

print("Baseline: ", mae(y_test, x_test[:, 4]))


# %%
