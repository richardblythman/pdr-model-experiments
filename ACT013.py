# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler

# read eth prices
eth_data = pd.read_csv("ETH_5m_.csv")
data = eth_data.set_index("timestamp")


# make future values as targets for the current smaples

data[["dir_open", "dir_high", "dir_low", "dir_close", "dir_volume"]] = data[
    ["open", "high", "low", "close", "volume"]
].diff()

data["dir_close"] = data["dir_close"].shift(-1)
data = data.dropna()

y = np.sign(data["dir_close"].values)
y[y <= 0] = 0
x = data.drop(["dir_close"], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
)


from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.backend import clear_session
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# %%
clear_session()
inputs = Input(shape=(x_train.shape[1],))
x = Dense(100, activation="relu")(inputs)
x = Dense(50, activation="relu")(x)
x = Dense(20, activation="relu")(x)
outputs = Dense(2, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=CategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["Accuracy"],
)

history = model.fit(
    x_train, y_train, batch_size=10000, epochs=10000, validation_split=0.2
)
model.evaluate(x_test, y_test)
