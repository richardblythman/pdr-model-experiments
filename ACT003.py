#%%
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV, Lasso
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

eth = yf.Ticker("ETH-USD")
btc = yf.Ticker("BTC-USD")
eth_data = eth.history(interval="60m", period="180d")
eth_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

data = eth_data

# Add technical indicators
data["MA_10"] = data["Close"].rolling(window=10).mean()
data["MA_30"] = data["Close"].rolling(window=30).mean()
data["MA_60"] = data["Close"].rolling(window=60).mean()
data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
data["Upper"], data["Middle"], data["Lower"] = (
    ta.volatility.BollingerBands(
        data["Close"], window=20, window_dev=2
    ).bollinger_hband(),
    ta.volatility.BollingerBands(
        data["Close"], window=20, window_dev=2
    ).bollinger_mavg(),
    ta.volatility.BollingerBands(
        data["Close"], window=20, window_dev=2
    ).bollinger_lband(),
)

data["y"] = data["Close"].shift(-1)
data.dropna(inplace=True)
t_1 = data["Close"]
print(data[["y", "Open", "Close"]])
print("MAE", mae(np.array(data["Open"]), np.array(data["Close"])))
print("MAE", mae(np.array(data["y"]), np.array(data["Close"])))

y = data["y"]
x = data.drop(["y"], axis=1).values


min_train_size = 1 * 24 * 170
x_train = x[0:min_train_size, :]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = y[0:min_train_size]

svr = SVR(kernel="rbf")
grid_params = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
}
sr = GridSearchCV(svr, param_grid=grid_params, n_jobs=-1, refit=True)
svr_params = sr.fit(x_train, y_train).best_params_
print(svr_params)


yhat_lr = []
yhat_lasso = []
yhat_bl = []
yhat_svr = []
y_test_full = []
print(min_train_size - x.shape[0])
for n in range(min_train_size, x.shape[0]):
    print(x.shape[0] - n)

    x_train = x[0:n, :]
    y_train = y[0:n]

    x_test = x[[n], :]
    y_test = y[n]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # LN prediction
    model = LinearRegression()
    model.fit(x_train, y_train)
    yhat_lr.append(model.predict(x_test)[0])

    # baseline
    yhat_bl.append(y_train[-1])

    # Lasso with CV
    if n == min_train_size:
        model = LassoCV(cv=5).fit(x_train, np.ravel(y_train))
        yhat_lasso.append(model.predict(x_test))
        alpha = model.alpha_
        print("cv")
    else:
        model = Lasso(alpha=alpha, max_iter=2000).fit(x_train, np.ravel(y_train))
        yhat_lasso.append(model.predict(x_test))

    # SVR after CV
    model = SVR(kernel="rbf", C=svr_params["C"]).fit(x_train, y_train)
    yhat_svr.append(model.predict(x_test))

    # global y_test
    y_test_full.append(y_test)

MAE_LR = mae(np.array(yhat_lr), np.array(y_test_full))
MAE_LASSO = mae(np.array(yhat_lasso), np.array(y_test_full))
MAE_SVR = mae(np.array(yhat_svr), np.array(y_test_full))
MAE_BL = mae(np.array(yhat_bl), np.array(y_test_full))


print("MAE LR", MAE_LR)
print("MAE LASSO", MAE_LASSO)
print("MAE SVR", MAE_SVR)
print("MAE BL", MAE_BL)

# MAE LR 5.203441003511898
# MAE LASSO 5.541768567853069
# MAE SVR 5.793383300850355
# MAE BL 5.194965069110577

#%%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
models = ["LR", "LASSO", "SVR", "Baseline"]
MAE = [MAE_LR, MAE_LASSO, MAE_SVR, MAE_BL]
ax.bar(models, MAE)
ax.set_ylim([7.5, 9])
ax.set_ylabel("MAE - Mean Absolute Error")
plt.show()

# %%
