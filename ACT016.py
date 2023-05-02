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

btc_data = pd.read_csv("BTC_1h_.csv")
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
data = data.dropna()


x = data.drop(["y", "return"], axis=1).values
r = data["return"].values
y = data["y"].values
y[y <= 0] = 0

model = Pipeline(
    steps=[
        ("scaller", MaxAbsScaler()),
        ("Predictor", LogisticRegression(max_iter=1000)),
    ]
)

nf = 5
kf = KFold(n_splits=nf)
pred_returns = np.zeros((nf,))
opt_returns = np.zeros((nf,))
timeframes = np.zeros((nf,))

for i, (ind_train, ind_test) in enumerate(kf.split(x)):
    model.fit(x[ind_train, :], y[ind_train])
    yhat = model.predict(x[ind_test, :])
    pred_returns[i] = np.sum(r[ind_test][yhat == 1])
    opt_returns[i] = np.sum(r[ind_test][y[ind_test] == 1])
    timeframes[i] = ind_test.shape[0]


results = pd.DataFrame(
    {
        "predictions": pred_returns,
        "ground_truth": opt_returns,
        "num_intervals": timeframes,
    }
)

results.to_csv("ACT016_BTC_1h_Trade_results.csv", index=False)
print(pred_returns)
print(opt_returns)
print(timeframes)

