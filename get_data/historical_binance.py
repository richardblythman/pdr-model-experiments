import ccxt
import pandas as pd

exchange = ccxt.binance()

symbols = ["ETH/USDT", "BTC/USDT"]
timeframe = "5m"
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

        start_time = int(
            (df.timestamp.iloc[-1] + pd.DateOffset(hours=1)).timestamp() * 1000
        )
        flag = df.timestamp.iloc[-1] <= pd.Timestamp(end_date, tz="UTC")

    data.to_csv(symbol[0:3] + "_" + timeframe + "_.csv", index=False)
