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

# get all the indicators
print("get naive standar indicators")
data = ta.add_all_ta_features(
    data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)


# make future values as targets for the current smaples
data[["dopen", "dhigh", "dlow", "dclose", "dvolume"]] = data[
    ["open", "high", "low", "close", "volume"]
].diff()
data["y_h"] = np.sign(data["dhigh"].shift(-1))
data["y_l"] = np.sign(data["dlow"].shift(-1))
data["y_c"] = np.sign(data["dclose"].shift(-1))
data = data.dropna()

labels_names = ["y_h", "y_l", "y_c"]
labels = data[labels_names]
predictors = data.drop(labels_names, axis=1)
scores = pd.DataFrame(["y_h", "y_l", "y_c"])

#
for label_name in labels_names:
    # set targets and predictors
    y = labels[label_name].values
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

    print("Fitting Logistic Regression for ", label_name)
    model = LogisticRegression()
    model = model.fit(x_train, y_train)
    yhat_logreg = model.predict(x_test)
    score_acc = acc(yhat_logreg, y_test)
    scores[label_name] = [score_acc, np.mean(y_test == 1), np.mean(y_test == -1)]


print("full data labels dist", np.mean(y == 1), np.mean(y == -1))

# %%
