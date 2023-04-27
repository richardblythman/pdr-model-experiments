import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV, Lasso

start_date = (
    pd.Timestamp.now(tz=None).floor("D")
    - pd.DateOffset(years=5)
    + pd.DateOffset(months=1)
)
eth = yf.Ticker("ETH-USD")
btc = yf.Ticker("BTC-USD")
eth_data = eth.history(interval="1m", period="7d")

# eth_data = eth.history(interval="1d", start=start_date)
# eth_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
# btc_data = btc.history(interval="1d", start=start_date)
# btc_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

data = pd.DataFrame({"eth_price": eth_data["Close"]})

x = sliding_window_view(data.values, 1 * 60 + 1, axis=0)
y = x[:, 0, -1:]
x = x[:, :, 0:-1]

min_train_size = 60 * 24 * 6
yhat_lr = []
yhat_lasso = []
yhat_bl = []

y_test_full = []
for n in range(min_train_size, x.shape[0]):
    x_train = x[0:n, 0, :]
    y_train = y[0:n]

    x_test = x[[n], 0, :]
    y_test = y[n]

    # LN prediction
    model = LinearRegression()
    model.fit(x_train, y_train)
    yhat_lr.append(model.predict(x_test)[0])

    # baseline
    yhat_bl.append(x_test[-1, -1])

    # Lasso with CV
    if n == min_train_size:
        model = LassoCV(cv=5).fit(x_train, np.ravel(y_train))
        yhat_lasso.append(model.predict(x_test))
        alpha = model.alpha_
        print("cv")
    else:
        model = Lasso(alpha=alpha).fit(x_train, np.ravel(y_train))
        yhat_lasso.append(model.predict(x_test))
    print(x.shape[0] - n)
    # global y_test
    y_test_full.append(y_test)


print("MAE LR", mae(np.array(yhat_lr), np.array(y_test_full)))
print("MAE LASSO", mae(np.array(yhat_lasso), np.array(y_test_full)))
print("MAE BL", mae(np.array(yhat_bl), np.array(y_test_full)))

# MAE LR 0.7091294143928014
# MAE LASSO 0.7048166443534697
# MAE BL 0.7026641970604536

#%%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
models = ["LR", "LASSO", "Baseline"]
MAE = [0.7091294143928014, 0.7048166443534697, 0.7026641970604536]
ax.bar(models, MAE)
ax.set_ylim([0.7, 0.71])
plt.show()
# %%
