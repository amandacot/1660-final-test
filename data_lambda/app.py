# -------------------------------
# Imports
# -------------------------------
import yfinance as yf
import pandas as pd
import requests
from functools import reduce
from fredapi import Fred
from datetime import datetime, timedelta
import boto3
import os
from zoneinfo import ZoneInfo  

def lambda_handler(event,context):
    # Current date in US/Eastern timezone as a datetime object
    today_dt = datetime.now(ZoneInfo('US/Eastern')).date()  

    # Today as a string in YYYY-MM-DD format
    today_str = today_dt.strftime('%Y-%m-%d')

    tomorrow_dt = today_dt + timedelta(days=1)           
    tomorrow_str = tomorrow_dt.strftime('%Y-%m-%d') 
# -------------------------------
# Configuration
# -------------------------------
    S3_BUCKET = "amazon-sagemaker-736116164611-us-east-2-ayappcqas719d5"  
    S3_KEY = "data/train19e.csv"


    FRED_API_KEY = "739a209da61bdd98f676253949fe9183"

    tickers = ["NVDA", "AMD", "TSM", "^GSPC", "^VIX",'SOXX','SMH']

# -------------------------------
# Step 1: Download stock/index data
# -------------------------------
    dfs = []
    for t in tickers:
        df = yf.download(t, start="2019-11-28", end=tomorrow_str, interval="1d")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = [f'Open_{t}', f'High_{t}', f'Low_{t}', f'Close_{t}', f'Volume_{t}']
        df = df.reset_index()  # Date becomes a column
        dfs.append(df)

# Merge all data on Date
    data = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), dfs)
    data = data.sort_values('Date').reset_index(drop=True)
    data = data.ffill()  # fill missing values

# futures of s and p 

    es = yf.Ticker("ES=F")
    es_hist = es.history(start="2019-11-28", end=tomorrow_str, interval="1d")[['Open','High','Low','Close','Volume']]
    es_hist = es_hist.reset_index()
    es_hist.columns = ['Date','ES_Open','ES_High','ES_Low','ES_Close','ES_Volume']
    es_hist['Date'] = es_hist['Date'].dt.tz_localize(None)
    data = pd.merge(data, es_hist, on='Date', how='left')
    data = data.ffill()

# -------------------------------
# Step 2: Add US 10-year Treasury yield (last 3 years)
# -------------------------------
    fred = Fred(api_key=FRED_API_KEY)
    us10y = fred.get_series('DGS10')
    us10y_df = us10y.reset_index()
    us10y_df.columns = ['Date', 'US10Y']

    three_years_ago = datetime.today() - timedelta(days=5*365)
    us10y_df = us10y_df[us10y_df['Date'] >= three_years_ago]
    us10y_df['US10Y'] = us10y_df['US10Y'].ffill()

    data = pd.merge(data, us10y_df, on='Date', how='left')

# -------------------------------
# Step 3: Technical Indicators for NVDA
# -------------------------------
# EMA
    data['EMA_12_NVDA'] = data['Close_NVDA'].ewm(span=12, adjust=False).mean()
    data['EMA_26_NVDA'] = data['Close_NVDA'].ewm(span=26, adjust=False).mean()

# MACD
    data['MACD_NVDA'] = data['EMA_12_NVDA'] - data['EMA_26_NVDA']
    data['MACD_signal_NVDA'] = data['MACD_NVDA'].ewm(span=9, adjust=False).mean()

# RSI
    delta = data['Close_NVDA'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI_NVDA'] = 100 - (100 / (1 + rs))

# ATR
    high_low = data['High_NVDA'] - data['Low_NVDA']
    high_close = (data['High_NVDA'] - data['Close_NVDA'].shift()).abs()
    low_close = (data['Low_NVDA'] - data['Close_NVDA'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_NVDA'] = tr.rolling(14).mean()

# OBV
    obv = [0]
    for i in range(1, len(data)):
        if data['Close_NVDA'].iloc[i] > data['Close_NVDA'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume_NVDA'].iloc[i])
        elif data['Close_NVDA'].iloc[i] < data['Close_NVDA'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume_NVDA'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV_NVDA'] = obv

    data['ES_Overnight_gap'] = data['ES_Open'] - data['ES_Close'].shift(1)
    data['ES_HighLow'] = data['ES_High'] - data['ES_Low']
    data['ES_vol_5d_Close'] = data['ES_Close'].pct_change().rolling(5).std().fillna(0)
    data['ES_vol_5d_Open'] = data['ES_Open'].pct_change().rolling(5).std().fillna(0)

    data['NVDA_Overnight_gap'] = data['Open_NVDA'] - data['Close_NVDA'].shift(1)
    data['NVDA_Close_diff'] = data['Close_NVDA'] - data['Close_NVDA'].shift(1)
    data['NVDA_HighLow'] = data['High_NVDA'] - data['Low_NVDA']


# -------------------------------
# CNN Fear & Greed Index
# -------------------------------

# Fetch data
    url = "https://api.alternative.me/fng/?limit=0"
    response = requests.get(url)
    fg_data = response.json()["data"]

# Convert to DataFrame
    fg_df = pd.DataFrame(fg_data)
    fg_df['Date'] = pd.to_datetime(fg_df['timestamp'], unit='s')
    fg_df = fg_df.sort_values('Date')
    fg_df = fg_df.rename(columns={"value": "fear_greed_index"})
    fg_df = fg_df[['Date', 'fear_greed_index']]

# Keep only last 3 years
    three_years_ago = datetime.today() - timedelta(days=5*365)
    fg_df = fg_df[fg_df['Date'] >= three_years_ago]

# Forward-fill missing values
    fg_df['fear_greed_index'] = fg_df['fear_greed_index'].ffill()

# Merge with main dataset
    data = pd.merge(data, fg_df, on='Date', how='left')
# -----
# Short-term (1 month)
    window_short = 21
    mean_short = data['Close_NVDA'].rolling(window_short).mean()
    std_short = data['Close_NVDA'].rolling(window_short).std()
    data['NVDA_zscore_1m_Close'] = (data['Close_NVDA'] - mean_short) / std_short
    data['NVDA_HILO_1m_Close'] = (data['NVDA_zscore_1m_Close'] > 2).astype(int)

# Long-term (1 year)
    window_long = 252
    mean_long = data['Close_NVDA'].rolling(window_long).mean()
    std_long = data['Close_NVDA'].rolling(window_long).std()
    data['NVDA_zscore_1y_Close'] = (data['Close_NVDA'] - mean_long) / std_long
    data['NVDA_HILO_1y_Close'] = (data['NVDA_zscore_1y_Close'] > 2).astype(int)

# Short-term (1 month)
    window_short = 21
    mean_short = data['Open_NVDA'].rolling(window_short).mean()
    std_short = data['Open_NVDA'].rolling(window_short).std()
    data['NVDA_zscore_1m_Open'] = (data['Open_NVDA'] - mean_short) / std_short
    data['NVDA_HILO_1m_Open'] = (data['NVDA_zscore_1m_Open'] > 2).astype(int)

# Long-term (1 year)
    window_long = 252
    mean_long = data['Open_NVDA'].rolling(window_long).mean()
    std_long = data['Open_NVDA'].rolling(window_long).std()
    data['NVDA_zscore_1y_Open'] = (data['Open_NVDA'] - mean_long) / std_long
    data['NVDA_HILO_1y_Open'] = (data['NVDA_zscore_1y_Open'] > 2).astype(int)


#---------- Broader market (No of stocks going up) long term trends



    sp500 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'UNH', 'JNJ','V', 'PG', 'XOM', 'CVX', 'HD', 'KO', 'PEP', 'ABBV', 'LLY', 'MRK','BAC', 'MA', 'DIS', 'PFE', 'AVGO', 'NFLX', 'CRM', 'ADBE', 'COST', 'NKE']


    prices = yf.download(sp500, start="2019-11-28", end=tomorrow_str)['Close']


# 52-week rolling mean and std for z-score
    rolling_mean = prices.rolling(21).mean()
    rolling_std  = prices.rolling(21).std()

# Compute z-score relative to 52-week mean
    z_scores = (prices - rolling_mean) / rolling_std

# Threshold: consider "close to high" if z > 1, "close to low" if z < -1
    new_highs = (z_scores > 1).sum(axis=1)  # count tickers close to 52-week high
    new_lows  = (z_scores < -1).sum(axis=1) # count tickers close to 52-week low

# HILO index
    hilo = new_highs - new_lows

# Make sure the index is a column
    hilo = hilo.reset_index()
    hilo.columns = ['Date', 'HILO_close']

# Merge with your main data
    data = pd.merge(data, hilo, on='Date', how='left')

# Fill missing values if needed
    data['HILO_close'] = data['HILO_close'].fillna(0)

#--- open 

    prices = yf.download(sp500, start="2019-11-28", end=tomorrow_str)['Open']


# 52-week rolling mean and std for z-score
    rolling_mean = prices.rolling(21).mean()
    rolling_std  = prices.rolling(21).std()

# Compute z-score relative to 52-week mean
    z_scores = (prices - rolling_mean) / rolling_std

# Threshold: consider "close to high" if z > 1, "close to low" if z < -1
    new_highs = (z_scores > 1).sum(axis=1)  # count tickers close to 52-week high
    new_lows  = (z_scores < -1).sum(axis=1) # count tickers close to 52-week low

# HILO index
    hilo = new_highs - new_lows

# Make sure the index is a column
    hilo = hilo.reset_index()
    hilo.columns = ['Date', 'HILO_open']

# Merge with your main data
    data = pd.merge(data, hilo, on='Date', how='left')

# Fill missing values if needed
    data['HILO_open'] = data['HILO_open'].fillna(0)


# Create relative features
    data['NVDA_vs_SP500_C'] = data['Close_NVDA'] / data['Close_^GSPC']
    data['NVDA_vs_TSM_C'] = data['Close_NVDA'] / data['Close_TSM']
    data['NVDA_vs_AMD_C'] = data['Close_NVDA'] / data['Close_AMD']

    data['NVDA_vs_SP500_O'] = data['Open_NVDA'] / data['Open_^GSPC']
    data['NVDA_vs_TSM_O'] = data['Open_NVDA'] / data['Open_TSM']
    data['NVDA_vs_AMD_O'] = data['Open_NVDA'] / data['Open_AMD']


    data["NVDA_vs_SOXX_C"] = data["Close_NVDA"] / data["Close_SOXX"]
    data["NVDA_vs_SMH_C"]  = data["Close_NVDA"] / data["Close_SMH"]


    data["NVDA_vs_SOXX_O"] = data["Open_NVDA"] / data["Open_SOXX"]
    data["NVDA_vs_SMH_O"]  = data["Open_NVDA"] / data["Open_SMH"]

    data["SOXX_ret1_O"] = data["Close_SOXX"].pct_change()
    data["SMH_ret1_O"]  = data["Close_SMH"].pct_change()


    data["Open_NVDA_change"] = data["Open_NVDA"].pct_change()
    data["Close_NVDA_change"]  = data["Close_NVDA"].pct_change()

# -------------------------------
# Step 4: Lagged Features (1-5 days)
# -------------------------------

    data = data.fillna(0)

    lag_days = [1,2,3,4,5]
    cols_to_lag = ['Open_NVDA','Close_NVDA', 'Volume_NVDA', 'EMA_12_NVDA', 'EMA_26_NVDA', 
                'MACD_NVDA', 'MACD_signal_NVDA', 'RSI_NVDA', 'ATR_NVDA', 
                'OBV_NVDA', 'US10Y','Open_TSM','Close_TSM','Open_AMD','Close_AMD','Open_^GSPC','Close_^GSPC','NVDA_HighLow','NVDA_Close_diff','NVDA_Overnight_gap','NVDA_vs_SP500_C','NVDA_vs_TSM_C','NVDA_vs_AMD_C','NVDA_vs_SP500_O','NVDA_vs_TSM_O','NVDA_vs_AMD_O']

    for col in cols_to_lag:
        for lag in lag_days:
            data[f"{col}_lag{lag}"] = data[col].shift(lag).fillna(0)



# Create additional ratios for lagged values
    for lag in [1,2,3,4,5]:
        data[f'NVDA_lag{lag}_vs_SP500_lag_C{lag}'] = data[f'Close_NVDA_lag{lag}'] / data[f'Close_^GSPC_lag{lag}']
        data[f'NVDA_lag{lag}_vs_TSM_lag_C{lag}'] = data[f'Close_NVDA_lag{lag}'] / data[f'Close_TSM_lag{lag}']
        data[f'NVDA_lag{lag}_vs_SP500_lag_O{lag}'] = data[f'Close_NVDA_lag{lag}'] / data[f'Close_^GSPC_lag{lag}']
        data[f'NVDA_lag{lag}_vs_TSM_lag_O{lag}'] = data[f'Close_NVDA_lag{lag}'] / data[f'Close_TSM_lag{lag}']

# -------------------------------
# Step 5: Save CSV locally
# -------------------------------
    if 'Volume_^VIX' in data.columns:
        data = data.drop(columns=['Volume_^VIX'])

    data['Target_Open_NVDA'] = data['Open_NVDA'].shift(-1)
#data.dropna(subset=['Target_Open_NVDA'], inplace=True)
    data = data.iloc[252:].reset_index(drop=True)
    data.to_csv('/tmp/train19e.csv', index=False)


# -------------------------------
# Step 6: Upload to S3 (optional)
# -------------------------------

    s3 = boto3.client('s3')
    s3.upload_file('/tmp/train19e.csv', S3_BUCKET, S3_KEY)
    return {"status": "success"}

