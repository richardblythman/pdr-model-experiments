import copy
# import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import ta
    
def get_ta_features(df):  
    df = ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )
    return df
    
def get_time_features(df):
    df = df.assign(sin_month=np.zeros(len(df)), cos_month=np.zeros(len(df)), sin_day=np.zeros(len(df)), cos_day=np.zeros(len(df)), sin_hour=np.zeros(len(df)), cos_hour=np.zeros(len(df)), sin_minute=np.zeros(len(df)), cos_minute=np.zeros(len(df)),)
    time_features = np.zeros((len(df),8))

    for i in range(len(time_features)):
        datetime = pd.to_datetime(df['datetime'][i], utc=True)
        time_features[i,0] = (np.sin(2 * np.pi * datetime.month/12))
        time_features[i,1] = (np.cos(2 * np.pi * datetime.month/12))
        time_features[i,2] = (np.sin(2 * np.pi * datetime.day/31))
        time_features[i,3] = (np.cos(2 * np.pi * datetime.day/31))
        time_features[i,4] = (np.sin(2 * np.pi * datetime.hour/24))
        time_features[i,5] = (np.cos(2 * np.pi * datetime.hour/24))
        time_features[i,6] = (np.sin(2 * np.pi * datetime.minute/60))
        time_features[i,7] = (np.cos(2 * np.pi * datetime.minute/60))

    df[['sin_month','cos_minute', 'sin_day', 'cos_day' , 'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute']] = time_features
    
    return df

def predict(df):
    # Load model
    model_name = "eth_5m"
    with open(f"weights/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)    

    predict = model.predict(df.iloc[-1]) # predict from last row
    pred = predict.argmax(axis=1)[0]
    conf = predict[:,1][0] 
    
    return pred, conf

if __name__ == '__main__':
    # df_candles should have columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]    
    df_candles = pd.read_csv("data/ETH_5m_candles.csv")
    
    print("Getting technical analysis features...")
    df = get_ta_features(df_candles)
    print("Getting time features...")
    df = get_time_features(df)
    
    df = df.dropna()
    df = df.drop(["timestamp", "datetime"], axis=1)
    
    print("Running predictions...")
    pred, conf = predict(df)
    print(f"Pred: {pred}, Conf: {conf}")