# %%
import requests
import pandas as pd

url = "https://predict-eth-data.oceanprotocol.com/merged"
start_date = "2018-01-01T00:00:00+00:00"
end_date = "2018-01-02T00:00:00+00:00"
params = {"start_date": start_date, end_date: end_date, "timeframe": "1h"}

response = requests.get(url, params=params)
#%%
data = response.json()
df = pd.DataFrame(data["data"])
df.to_csv("response.csv", index=False)
# %%
