import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def fetch_stock_data(tickers, start, end):
    dataset = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)
            info = stock.info

            #get relevant fundamental data
            fundamentals = {
                "Ticker": ticker,
                "Market Cap": info.get("marketCap", None),
                "PE Ratio": info.get("trailingPE", None),
                "PB Ratio": info.get("priceToBook", None),
                "EPS": info.get("trailingEps", None),
                "Debt-to-Equity": info.get("debtToEquity", None),
            }

            for date, row in hist.iterrows():
                dataset.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Open": row["Open"],
                    "High": row["High"],
                    "Low": row["Low"],
                    "Close": row["Close"],
                    "Volume": row["Volume"],
                    **fundamentals
                })
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return pd.DataFrame(dataset)

def fill_missing_dates(df, group_col="Ticker", date_col="Date"):
    all_dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    all_groups = df[group_col].unique()
    full_index = pd.MultiIndex.from_product([all_groups, all_dates], names=[group_col, date_col])
    df = df.set_index([group_col, date_col])
    full_df = df.reindex(full_index)

    #forward fill missing val
    full_df = full_df.groupby(level=group_col).ffill()
    full_df = full_df.reset_index()

    return full_df

def preprocess_data(df):
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"])
    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days

    # Standardize features
    numerical_cols = ["Open", "High", "Low", "Volume", "Market Cap", "PE Ratio", "PB Ratio", "EPS", "Debt-to-Equity"]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
