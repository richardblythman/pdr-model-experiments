# %%
import ccxt
import pandas as pd

exchange = ccxt.binance()

symbols = ["XRP/USDT"]
timeframe = "1h"
delta_time = pd.DateOffset(minutes=5)
start_date = "2017-09-01"
end_date = "2023-04-19"


for symbol in symbols:
    start_time = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    flag = True
    data = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    while flag:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_time,
            limit=1000,
            params={"until": end_time},
        )
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        data = pd.concat([data, df])

        start_time = int((df.timestamp.iloc[-1] + delta_time).timestamp() * 1000)

        flag = df.timestamp.iloc[-1] + delta_time <= pd.Timestamp(end_date, tz="UTC")

    data.to_csv(symbol[0:3] + "_" + timeframe + "_.csv", index=False)

# %%
