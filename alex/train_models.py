# %%
import sys

sys.path.append("../")
import ccxt
import os
import pandas as pd
import ta
import time
import pickle
import warnings
from models.OceanModel import OceanModel
from models.richard1 import RichardModel1


# Jamie's model 1
#model = OceanModel("binance", "btc/tusd", "5m")
#model.train_from_csv("./csvs/binance_btctusd_5m.csv")
#model.pickle_model("./trained_models/")


# richard
model = RichardModel1("binance",'btc-tusd','5m')
model.train_from_csv("./csvs/binance_btctusd_5m.csv")
model.pickle_model("./trained_models/")

# Jaime's model 2
model = OceanModel("binance", "btc/tusd", "5m")
model.train_from_csv("./csvs/binance_btctusd_5m.csv")
model.pickle_model("./trained_models/")


# %%
