# %%
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

# read eth prices
eth_data = pd.read_csv("ETH.csv")
eth_data = eth_data.set_index("timestamp")

# get all the indicators
print("get naively standar indicators")
data = ta.add_all_ta_features(
    eth_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)


# make future values as targets for the current smaples
data = eth_data
data["y"] = data["close"].shift(-1)
data.dropna(inplace=True)

# set targets and predictors
close = data["close"].values
y = data["y"].values
x = data.drop(["y"], axis=1).values

# %%
min_train_size = x.shape[0] - int(0.1 * x.shape[0])

yhat_lda = []
yhat_qda = []
y_test_full = []

x_train = x[0:min_train_size, :]
y_train = y[0:min_train_size]

x_test = x[min_train_size:, :]
y_test = y[min_train_size:]
close_test = close[min_train_size:]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Fitting OLS")
model = LinearRegression()
model = model.fit(x_train, y_train)
yhat_logreg = model.predict(x_test)
score = mae(yhat_logreg, y_test)
print("OLS", score)

print("Fitting Lasso")
model = LassoCV(cv=5)  # .fit(x_train, np.ravel(y_train))
model = model.fit(x_train, y_train)
yhat_logreg = model.predict(x_test)
score = mae(yhat_logreg, y_test)
print("Lasso", score)


print("Calculating baseline score")
print("OLS", mae(close_test, y_test))


# %%
