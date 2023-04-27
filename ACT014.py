# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler
from scipy import signal

# read eth prices
eth_data = pd.read_csv("ETH_5m_.csv")
data = eth_data.set_index("timestamp")


# make future values as targets for the current smaples

x = data[["open", "high", "low", "close", "volume"]].diff()
x["volume"] = x["volume"].shift(-2)
x = x.dropna()

s1 = x.volume.values
s2 = x.close.values
corr = signal.correlate(s1, s2)
lags = signal.correlation_lags(x.shape[0], x.shape[0])
corr = corr / np.max(corr)
print(np.sort(-corr)[0:5])
print(lags[np.argsort(-corr)[0:5]])
import matplotlib.pyplot as plt

plt.plot(lags, corr)


# %%
