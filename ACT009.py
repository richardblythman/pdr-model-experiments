# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import cohen_kappa_score as kappa
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
data = eth_data.set_index("timestamp")


# make future values as targets for the current smaples
data[["dopen", "dhigh", "dlow", "dvolume"]] = data[
    ["open", "high", "low", "volume"]
].diff()
data = data.dropna()

labels = data["close"]
predictors = data.drop(["close"], axis=1)

# set targets and predictors
y = labels.values
x = predictors.values

ind_zero = y != 0
y = y[ind_zero]
x = x[ind_zero, :]

# %
min_train_size = x.shape[0] - int(0.1 * x.shape[0])

yhat_lda = []
yhat_qda = []
y_test_full = []

x_train = x[0:min_train_size, :]
y_train = y[0:min_train_size]

x_test = x[min_train_size:, :]
y_test = y[min_train_size:]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Fitting OLS ")
model = LinearRegression()
model = model.fit(x_train, y_train)
yhat = model.predict(x_test)
score = mae(yhat, y_test)
z = scaler.inverse_transform(x_test)
print(
    score,
    mae(z[:, 0], y_test),
    mae(z[:, 1], y_test),
    mae(z[:, 2], y_test),
    mae(scaler.inverse_transform(x_train)[:, 0], y_train),
)


# %%
