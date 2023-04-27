# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc

# read eth prices
eth_data = pd.read_csv("ETH_1h_.csv")
data = eth_data.set_index("timestamp")


data[["dir_open", "dir_high", "dir_low", "dir_close"]] = np.sign(
    data[["open", "high", "low", "close"]].diff()
)

data = data.dropna()
y = data["dir_close"].values
y[y <= 0] = -1
x = data.drop(["dir_close","close"], axis=1).values
x[x <= 0] = -1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
mode = model.fit(x_train, y_train)
yhat = model.predict(x_test)
print("Acc: ", acc(yhat, y_test))
# %%
