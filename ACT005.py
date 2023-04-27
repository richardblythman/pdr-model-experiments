# %%
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier

eth = yf.Ticker("ETH-USD")
btc = yf.Ticker("BTC-USD")

start_date = "2020-01-01"
end_date = "2020-01-30"

eth_data = eth.history(interval="1h", period="2y")
# eth_data = eth.history(interval="1h", start=start_date, end=end_date)
eth_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

data = ta.add_all_ta_features(
    eth_data,
    open="Open",
    high="High",
    low="Low",
    close="Close",
    volume="Volume",
    fillna=True,
)


data["MA_02"] = eth_data["Close"].rolling(window=2).mean()
data["MA_06"] = eth_data["Close"].rolling(window=6).mean()
data["MA_10"] = eth_data["Close"].rolling(window=10).mean()
data["MA_30"] = eth_data["Close"].rolling(window=30).mean()
data["MA_60"] = eth_data["Close"].rolling(window=60).mean()


data["y"] = np.sign(data["Close"].diff().shift(-1))
data["delta_price"] = data["Close"].diff()
data["delta_dir"] = np.sign(data["delta_price"].diff())
r = [data["Close"].shift(i) for i in range(1, 13)]
r.append(data)
data = pd.concat(r, axis=1)
data.dropna(inplace=True)

prices = data["Close"].values

y = data["y"].values
x = data.drop(["y"], axis=1).values

min_train_size = x.shape[0] - int(0.5 * x.shape[0])

yhat_lda = []
yhat_qda = []
y_test_full = []

x_train = x[0:min_train_size, :]
y_train = y[0:min_train_size]

x_test = x[min_train_size:, :]
y_test = y[min_train_size:]
prices_test = prices[min_train_size:, 0]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LogisticRegression(max_iter=2000)
model = model.fit(x_train, y_train)
yhat_logreg = model.predict(x_test)
print("Logistic", model.score(x_test, y_test))

prof = 0
curr_prof = np.zeros((yhat_logreg.shape[0], 1))


for i in range(0, yhat_logreg.shape[0] - 1):
    if (yhat_logreg[i]) == 1:
        # buy
        curr_prof[i, 0] = (prices_test[i + 1] - prices_test[i]) / prices_test[i]
        prof = prof + curr_prof[i, 0]


n = 1000
rand_prof = np.zeros((n, 1))
curr_prof_rand = np.zeros((yhat_logreg.shape[0], n))
rand_pred = np.random.choice([-1, 1], size=(yhat_logreg.shape[0], n))

for j in range(0, n):
    for i in range(0, yhat_logreg.shape[0] - 1):
        if (rand_pred[i, j]) == 1:
            # buy
            curr_prof_rand[i, j] = (prices_test[i + 1] - prices_test[i]) / prices_test[
                i
            ]
            rand_prof[j] = rand_prof[j] + curr_prof_rand[i, j]


print("Profit", prof)
print("random", np.mean(rand_prof))

import matplotlib.pyplot as plt

z = plt.hist(rand_prof)

returns = np.cumsum(curr_prof) * 100
fig, ax1 = plt.subplots()

# Plot the first signal on the first axis
ax1.plot(prices_test, "b-")
ax1.set_xlabel("x")
ax1.set_ylabel("ETH-price", color="b")
ax1.tick_params("y", colors="b")

# Create a second axis sharing the same x-axis as the first axis
ax2 = ax1.twinx()

# Plot the second signal on the second axis
ax2.plot(returns, "r-")
ax2.set_ylabel("Returns (%)", color="r")
ax2.tick_params("y", colors="r")

plt.show()

# %%
# import copy

# acc = []
# y = copy.deepcopy(y_test[:])
# for i in range(0, 100000):
#     np.random.shuffle(y)
#     acc.append(np.mean(y == y_test))

# import matplotlib.pyplot as plt

# z = plt.hist(np.array(acc), bins=np.linspace(0.45, 0.55, 100))
