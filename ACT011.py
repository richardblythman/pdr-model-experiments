# %%
import pandas as pd
import numpy as np

# read eth prices
eth_data = pd.read_csv("ETH_1h_.csv")
data = eth_data.set_index("timestamp")


# make future values as targets for the current smaples

data["profit"] = data["close"].diff().shift(-1) / data["close"]
data = data.dropna()

n = 100000
probs = [0.45, 0.5, 0.55]
returns = np.repeat(np.expand_dims(data["profit"].values, axis=1), n, axis=1)

profit = np.zeros((n, len(probs)))
for i, p in enumerate(probs):
    noise = np.random.choice([0, 1], size=(data.shape[0], n), p=[p, 1 - p])
    labels = np.sign(returns)
    labels[labels <= 0] = 0

    ind = noise == 1
    labels[ind] = np.logical_not(labels[ind])
    profit[:, i] = np.mean(returns * labels.astype("float"), axis=0) * 100


profit = profit * 24 * 365
import matplotlib.pyplot as plt

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 8))

# Plot the data on each subplot
axs[0].hist(profit[:, 0])
axs[0].set_title(str(probs[0]))
axs[1].hist(profit[:, 1])
axs[1].set_title(str(probs[1]))
axs[2].hist(profit[:, 2])
axs[2].set_title(str(probs[2]))


# Add a main title for the figure
fig.suptitle("Six Subplots")

# Display the figure
plt.show()
plt.hist(profit)

# %%
