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
import lightgbm as lgb
import seaborn as sns

btc_data = pd.read_csv("data/crypto/BTC_1h_.csv")
btc_data = btc_data.set_index("timestamp")

print(btc_data.head())

btc_data.index.name = 'datetime'
btc_data['timestamp'] = pd.to_datetime(btc_data.index.values,utc=True)

for i in range(len(btc_data)):
    btc_data['timestamp'][i] = btc_data['timestamp'][i].timestamp() 

btc_data = ta.add_all_ta_features(
    btc_data,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)

# Time Features

btc_data = btc_data.assign(sin_month=np.zeros(len(btc_data)), cos_month=np.zeros(len(btc_data)),sin_day=np.zeros(len(btc_data)),cos_day=np.zeros(len(btc_data)),sin_hour=np.zeros(len(btc_data)),cos_hour=np.zeros(len(btc_data)),sin_minute=np.zeros(len(btc_data)),cos_minute=np.zeros(len(btc_data)),)

time_features = np.zeros((len(btc_data),8))

for i in range(len(time_features)):

    timestamp = pd.to_datetime(btc_data.index[i],utc=True)

    time_features[i,0] = (np.sin(2 * np.pi * timestamp.month/12))
    time_features[i,1] = (np.cos(2 * np.pi * timestamp.month/12))
    time_features[i,2] = (np.sin(2 * np.pi * timestamp.day/31))
    time_features[i,3] = (np.cos(2 * np.pi * timestamp.day/31))
    time_features[i,4] = (np.sin(2 * np.pi * timestamp.hour/24))
    time_features[i,5] = (np.cos(2 * np.pi * timestamp.hour/24))
    time_features[i,6] = (np.sin(2 * np.pi * timestamp.minute/60))
    time_features[i,7] = (np.cos(2 * np.pi * timestamp.minute/60))

btc_data[['sin_month','cos_minute', 'sin_day', 'cos_day' , 'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute']] = time_features

data = copy.deepcopy(btc_data)
data["return"] = data["close"].diff().shift(-1) / data["close"]
data["y"] = np.sign(data["close"].diff().shift(-1))
data = data.dropna()



x = data.drop(["y", "return", "timestamp"], axis=1)
r = data["return"]
y = data["y"]
y[y <= 0] = 0

model = Pipeline(
    steps=[
        ("scaller", MaxAbsScaler()),
        ("Predictor", LogisticRegression(max_iter=1000)),
    ]
)

features = list(x.columns )

nf = 5
kf = KFold(n_splits=nf)
pred_returns = np.zeros((nf,))
opt_returns = np.zeros((nf,))
timeframes = np.zeros((nf,))
scores = np.zeros((nf,))
importances = []

params = {
    'early_stopping_rounds': 50,
    'objective': 'multiclass',
    'num_class': 2,
    }

# from: https://blog.amedama.jp/entry/lightgbm-cv-feature-importance
# (used in nyanp's Optiver solution)
def plot_importance(importances, features_names = features, PLOT_TOP_N = 20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()

for i, (ind_train, ind_test) in enumerate(kf.split(x)):
    purge = 1
    print(i)
    ind_test = ind_test[purge:-purge]
    print(ind_test)
    train_dataset = lgb.Dataset(x.iloc[ind_train, :],
                                y[ind_train].values, 
                                feature_name = features, 
                               )
    val_dataset = lgb.Dataset(x.iloc[ind_test, :], 
                              y[ind_test].values, 
                              feature_name = features, 
                             )
    
    model = lgb.train(params = params,
                      train_set = train_dataset, 
                      valid_sets=[train_dataset, val_dataset],
                      valid_names=['tr', 'vl'],
                      num_boost_round = 5000,
                      verbose_eval = 100,     
                      # feval = correlation,
                     )
 
    importances.append(model.feature_importance(importance_type='gain'))


    y_pred = model.predict(x.iloc[ind_test, :])  
    y_pred = y_pred.argmax(axis=1)
    # oof_valid += list(   y.iloc[ind_test].values    )
    scores[i] = acc(y[ind_test], y_pred)
    
    pred_returns[i] = np.sum(r[ind_test][y_pred == 1]) / len(r[ind_test][y_pred == 1])
    opt_returns[i] = np.sum(r[ind_test][y[ind_test] == 1]) / len(r[ind_test][y_pred == 1])
    timeframes[i] = ind_test.shape[0]

plot_importance(np.array(importances), features, PLOT_TOP_N = 20, figsize=(10, 5))

results = pd.DataFrame(
    {
        "predictions": pred_returns,
        "ground_truth": opt_returns,
        "num_intervals": timeframes,
    }
)

results.to_csv("ACT017_BTC_1h_Trade_results.csv", index=False)
print(pred_returns)
print(opt_returns)
print(timeframes)

