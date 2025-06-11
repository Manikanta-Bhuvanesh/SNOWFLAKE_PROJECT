import yfinance as yf
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def fetch_single_stock(symbol,full_load) -> pd.DataFrame:
    """
    Fetch data for a single stock symbol with error handling.
    This function is designed to be used with multiprocessing.
    """
    suffixes = ['.NS', '.BO']
    if full_load ==True:
        period='max'
    else:
        period='5d'
    
    for suffix in suffixes:
        try:
            ticker_symbol = symbol + suffix
            ticker = yf.Ticker(ticker_symbol)
            
            # Fetch 1 day of 1-minute data for speed (change to 'max' for production)
            data = ticker.history(period=period, interval='1m')
            
            if data.empty:
                continue

            # Efficient column selection and renaming
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
                'Open': 'OPEN', 'High': 'HIGH', 'Low': 'LOW',
                'Close': 'CLOSE', 'Volume': 'VOLUME'
            })

            # Add metadata columns
            now = datetime.now()
            data['SYMBOL'] = ticker_symbol
            data['TICKER_TIME'] = data.index
            data['LAST_UPDATED_AT'] = now

            # Reorder columns and clean data
            data.reset_index(drop=True, inplace=True)
            data = data[['SYMBOL', 'TICKER_TIME', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'LAST_UPDATED_AT']]
            data = data.dropna(how='any')

            return data

        except Exception as e:
            continue

    return pd.DataFrame()


def fetch_stocks_parallel(symbols,full_load=False) -> pd.DataFrame:
    all_dataframes = []
    
    with ThreadPoolExecutor(max_workers=7) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(fetch_single_stock, symbol, full_load): symbol
            for symbol in symbols
        }
        # Process completed tasks
        for future in tqdm(as_completed(future_to_symbol), total=len(future_to_symbol), desc="Fetching stocks", unit="stock"):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if not df.empty:
                    all_dataframes.append(df)
                    
            except Exception as e:
                continue
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()