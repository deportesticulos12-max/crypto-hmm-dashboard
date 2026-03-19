import requests
import pandas as pd
import time

def check_binance_data(symbol="ETHUSDT", interval="1d", limit=3200):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    remaining = limit
    last_end_time = None
    
    print(f"Fetching {limit} bars for {symbol} {interval}...")
    
    while remaining > 0:
        fetch_count = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": fetch_count}
        if last_end_time:
            params["endTime"] = last_end_time - 1
            
        resp = requests.get(url, params=params)
        data = resp.json()
        
        if not data:
            break
            
        print(f"  Fetched {len(data)} bars. Earliest: {pd.to_datetime(data[0][0], unit='ms')}")
        all_data = data + all_data
        remaining -= len(data)
        last_end_time = data[0][0]
        
        if len(data) < fetch_count:
            break
            
    print(f"Total bars fetched: {len(all_data)}")
    return len(all_data)

if __name__ == "__main__":
    check_binance_data("ETHUSDT", "1d", 3200)
    check_binance_data("BTCUSDT", "1d", 3200)
