# %%
import ccxt
import pandas as pd
import requests

exchange = ccxt.binance()

symbols = ["LTCUSDT"]
timeframe = "5m"
delta_time = pd.DateOffset(minutes=5)
start_date = "2017-09-01"
end_date = "2023-04-19"

columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "base_asset",
    "quote_asset",
    "none",
]


for symbol in symbols:
    start_time = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    flag = True
    data = pd.DataFrame(columns=columns)
    while flag:
        payload = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": start_time,
            "limit": 1000,
            "endTime": end_time,
        }
        r = requests.get("https://api.binance.com/api/v3/klines", params=payload)
        values = r.json()
        df = pd.DataFrame(values, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        data = pd.concat([data, df])

        start_time = int((df.timestamp.iloc[-1] + delta_time).timestamp() * 1000)

        flag = df.timestamp.iloc[-1] + delta_time <= pd.Timestamp(end_date, tz="UTC")

    data.to_csv(symbol + "_" + timeframe + "_.csv", index=False)

# %%
