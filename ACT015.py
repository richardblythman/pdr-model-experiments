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

eth_data = pd.read_csv("ETH_1h_.csv")
btc_data = pd.read_csv("BTC_1h_.csv")
xrp_data = pd.read_csv("XRP_1h_.csv")
ltc_data = pd.read_csv("LTC_1h_.csv")
eth_data = eth_data.set_index("timestamp")
btc_data = btc_data.set_index("timestamp")
xrp_data = xrp_data.set_index("timestamp")
ltc_data = ltc_data.set_index("timestamp")


eth_data = ta.add_all_ta_features(
    eth_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)

btc_data = ta.add_all_ta_features(
    btc_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)

xrp_data = ta.add_all_ta_features(
    xrp_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)

ltc_data = ta.add_all_ta_features(
    ltc_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)


data = pd.concat([btc_data], axis=1, keys=["btc"])
# data = data.diff()
data["y"] = np.sign(data["btc"]["close"].diff().shift(-1))
data = data.dropna()


x = data.drop(["y"], axis=1).values
y = data["y_btc"].values
y[y <= 0] = 0

model = Pipeline(
    steps=[
        ("scaller", MaxAbsScaler()),
        ("Predictor", LogisticRegression(max_iter=1000)),
    ]
)
nf = 5
kf = KFold(n_splits=nf)
scores = np.zeros((nf,))

for i, (ind_train, ind_test) in enumerate(kf.split(x)):
    model.fit(x[ind_train, :], y[ind_train])
    scores[i] = model.score(x[ind_test, :], y[ind_test])
    yhat = model.predict(x[ind_test, :])
    scores2[i] = acc(yhat, yy[ind_test])

print(scores, np.mean(scores))
