import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
from datetime import datetime, timedelta,date
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import os,json

@dataclass
class TradingConfig:
    """Configuration class for trading parameters"""
    initial_capital: float = 100000
    min_data_points: int = 100
    transaction_cost: float = 0.002

config = TradingConfig()

def DataFetcher(symbol: str, interval: str = '1d', min_years: int = 2):
    tz = pytz.timezone('Asia/Kolkata')
    suffixes = ['.NS', '.BO']
    # start_date = (datetime.now(tz) - timedelta(days=min_years * 365)).strftime('%Y-%m-%d')
    
    for suffix in suffixes:
        try:
            ticker_symbol = symbol + suffix
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period='max', interval=interval)
            
            if data.empty or len(data) < config.min_data_points:
                continue
                
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                continue
            
            data = data.dropna()
            data = data[data['Volume'] > 0]
            data = data[(data['High'] >= data['Low']) & (data['High'] >= data['Close']) & 
                       (data['Low'] <= data['Close']) & (data['Open'] > 0)]
            
            if len(data) < config.min_data_points:
                continue
                
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'Volume'
            }, inplace=True)
            
            data['Time'] = data.index.date
            data.reset_index(drop=True, inplace=True)
            
            return data, ticker_symbol
            
        except Exception:
            continue
    
    return pd.DataFrame(), None

def rma(close, length):
    """Optimized RMA calculation - this one was already quite efficient"""
    alpha = 1 / length
    rma_values = np.zeros_like(close, dtype=float)
    rma_values[length - 1] = np.mean(close[:length])
    
    for i in range(length, len(close)):
        rma_values[i] = alpha * close[i] + (1 - alpha) * rma_values[i - 1]
    
    return rma_values

def calculate_wma(values, length):
    """Vectorized WMA calculation using numpy operations"""
    values = np.array(values)
    n = len(values)
    wma_values = np.full(n, np.nan)
    
    if n < length:
        return wma_values.tolist()
    
    # Pre-calculate weights
    weights = np.arange(1, length + 1)
    weights_sum = weights.sum()
    
    # Vectorized calculation for all valid windows
    for i in range(length - 1, n):
        window = values[i - length + 1:i + 1]
        wma_values[i] = np.dot(window, weights) / weights_sum
    
    return wma_values.tolist()

def calculate_wma(values, length):
    """Even more optimized WMA using sliding window approach"""
    values = np.array(values)
    n = len(values)
    
    if n < length:
        return [np.nan] * n
    
    # Create sliding windows using numpy stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Get all windows at once
    windows = sliding_window_view(values, window_shape=length)
    
    # Pre-calculate weights
    weights = np.arange(1, length + 1)
    weights_sum = weights.sum()
    
    # Calculate WMA for all windows at once
    wma_results = np.sum(windows * weights, axis=1) / weights_sum
    
    # Pad with NaN for initial values
    result = np.full(n, np.nan)
    result[length-1:] = wma_results
    
    return result.tolist()

def calculate_supertrend(df, length=10, factor=3.0, ma_type='WMA'):
    """Optimized SuperTrend calculation with vectorized operations"""
    df = df.reset_index(drop=True).copy()
    
    # Extract arrays for vectorized operations
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['Volume'].values
    
    # Vectorized VWMA calculation
    cv = close * volume
    cv_wma = np.array(calculate_wma(cv, length))
    v_wma = np.array(calculate_wma(volume, length))
    
    # Handle division by zero
    vwma = np.divide(cv_wma, v_wma, out=np.zeros_like(cv_wma), where=v_wma!=0)
    
    # Vectorized True Range calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    # Handle first row where prev_close is NaN
    custom_true_range = np.where(np.isnan(prev_close), tr1, np.maximum(np.maximum(tr1, tr2), tr3))
    
    # Calculate ATR
    atr = rma(custom_true_range, length)
    
    # Calculate initial bands
    upper_band = vwma + factor * atr
    lower_band = vwma - factor * atr
    
    # Vectorized band adjustment
    n = len(df)
    prev_upper_band = np.roll(upper_band, 1)
    prev_lower_band = np.roll(lower_band, 1)
    prev_close_shift = np.roll(close, 1)
    
    # Initialize arrays for results
    final_upper_band = upper_band.copy()
    final_lower_band = lower_band.copy()
    direction = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    
    # Process band adjustments and direction calculation
    for i in range(1, n):
        # Adjust lower band
        if not (lower_band[i] > prev_lower_band[i] or prev_close_shift[i] < prev_lower_band[i]):
            final_lower_band[i] = prev_lower_band[i]
        
        # Adjust upper band
        if not (upper_band[i] < prev_upper_band[i] or prev_close_shift[i] > prev_upper_band[i]):
            final_upper_band[i] = prev_upper_band[i]
        
        # Calculate direction
        if np.isnan(atr[i-1]):
            direction[i] = 1
        elif i > 0 and supertrend[i-1] == prev_upper_band[i]:
            direction[i] = -1 if close[i] > final_upper_band[i] else 1
        else:
            direction[i] = 1 if close[i] < final_lower_band[i] else -1
        
        # Calculate supertrend
        supertrend[i] = final_lower_band[i] if direction[i] == -1 else final_upper_band[i]
    
    # Assign results back to dataframe
    df['atr'] = atr
    df['upperBand'] = final_upper_band
    df['lowerBand'] = final_lower_band
    df['direction'] = direction
    df['superTrend'] = supertrend
    
    # Remove NaN rows
    df.dropna(inplace=True)
    
    return df

def price_sp_wma(df):
    """Calculate WMAs for price and supertrend"""
    df['price_WMA'] = calculate_wma(df['close'].values, 20)
    df['superTrend_WMA'] = calculate_wma(df['superTrend'].values, 100)
    df.dropna(inplace=True)
    return df

def calculate_data_point(df, n):
    """Calculate single data point (matches original)"""
    data_points = []
    label = []
    for i in range(len(df)-1, len(df) - 1 - n, -1):
        data_points.append(df['superTrend'].iloc[i])
        label_i = 1 if df['price_WMA'].iloc[i] > df['superTrend_WMA'].iloc[i] else 0
        label.append(label_i)
    return data_points, label

def calculate_data_points(df, window_size=10):
    """Calculate all data points (matches original)"""
    data = []
    labels = []
    for i in range(window_size, len(df)):
        data_point, label = calculate_data_point(df[i-window_size:i], window_size)
        data.append(data_point)
        labels.append(label)
    return np.array(data), np.array(labels)

def knn_weighted_series(data, labels, k, x):
    """Original KNN implementation"""
    distances = np.abs(data - x)
    sorted_indices = distances.argsort()
    nearest_indices = sorted_indices[:k]
    weights = 1 / (distances[nearest_indices] + 1e-6)
    weighted_labels = weights * labels[nearest_indices]
    weighted_sum = weighted_labels.sum()
    total_weight = weights.sum()
    return weighted_sum / total_weight if total_weight else 0

def apply_corrected_trading_logic(df):
    """Original trading logic implementation"""
    df = df.reset_index(drop=True)
    last_signal = 'none'
    signals = ['none'] * len(df)
    
    for i in range(1, len(df)):
        if last_signal != 'long' and ((df.loc[i, 'label_'] == 1 and (df.loc[i-1, 'label_'] != 1 or df.loc[i-1, 'label_'] not in [1,0])) or 
                                   (df.loc[i, 'direction'] == -1 and df.loc[i-1, 'direction'] == 1 and df.loc[i, 'label_'] == 1)):
            signals[i] = 'Buy'
            last_signal = 'long'
        elif last_signal == 'long' and ((df.loc[i, 'close'] < df.loc[i, 'longTrailingStop']) or 
                                       (df.loc[i, 'label_'] == 1 and df.loc[i, 'direction'] == 1) or 
                                       (df.loc[i, 'label_'] == 0 and df.loc[i, 'direction'] == -1)):
            signals[i] = 'Sell'
            last_signal = 'none'
    
    df['signal'] = signals
    return df


def back_test_metrice(df: pd.DataFrame):
    """
    Evaluates completed trades (Buy→Sell pairs) with:
    - Trade accuracy (Hit Rate)
    - Profitability (Profit Factor, Win/Loss Ratio)
    - Risk metrics (Drawdown, Sharpe/Sortino)
    - Benchmark comparison (vs. Buy & Hold)
    - Trade analysis (MFE/MAE)
    - Annualized return percentage
    - Average holding days per trade
    """
    if df.empty or 'signal' not in df.columns:
        return {"error": "Invalid DataFrame"}

    # --- Step 1: Identify Completed Trades ---
    trades = []
    current_trade = None
    equity = [100]  # Starting equity (100%)

    for idx, row in df.iterrows():
        # Enter trade on Buy signal
        if row['signal'] == 'Buy' and current_trade is None:
            current_trade = {
                'entry_time': row['Time'],
                'entry_price': row['close'],
                'exit_time': None,
                'exit_price': None,
                'max_gain_pct': 0,  # MFE (Maximum Favorable Excursion)
                'max_loss_pct': 0,   # MAE (Maximum Adverse Excursion)
                'shares': 1  # Position sizing (adjust if needed)
            }
        
        # Update MFE/MAE for active trade
        elif current_trade is not None:
            current_price = row['close']
            gain_pct = (current_price - current_trade['entry_price']) / current_trade['entry_price'] * 100
            current_trade['max_gain_pct'] = max(current_trade['max_gain_pct'], gain_pct)
            current_trade['max_loss_pct'] = min(current_trade['max_loss_pct'], gain_pct)
            
            # Exit trade on Sell signal
            if row['signal'] == 'Sell':
                current_trade['exit_time'] = row['Time']
                current_trade['exit_price'] = current_price
                current_trade['profit_pct'] = gain_pct
                current_trade['holding_days'] = (current_trade['exit_time'] - current_trade['entry_time']).days
                trades.append(current_trade)
                
                # Update equity curve
                equity.append(round(equity[-1] * (1 + gain_pct/100),2))
                current_trade = None
    
    # Handle unrealized trades (closed at last price)
    if current_trade is not None:
        current_trade['exit_time'] = df['Time'].iloc[-1]
        current_trade['exit_price'] = df['close'].iloc[-1]
        current_trade['profit_pct'] = (current_trade['exit_price'] - current_trade['entry_price']) / current_trade['entry_price'] * 100
        current_trade['holding_days'] = (current_trade['exit_time'] - current_trade['entry_time']).days
        trades.append(current_trade)
        equity.append(round(float(equity[-1] * (1 + current_trade['profit_pct']/100)),2))

    if not trades:
        return {"error": "No completed trades found"}

    # --- Step 2: Calculate Metrics ---
    winning_trades = [t for t in trades if t['profit_pct'] > 0]
    losing_trades = [t for t in trades if t['profit_pct'] <= 0]
    
    total_trades = len(trades)
    hit_rate = len(winning_trades) / total_trades * 100
    avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
    max_win = np.max([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
    max_loss = np.min([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
    expected_value = (hit_rate * avg_win + (100 - hit_rate) * avg_loss) / 100
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Profit Factor
    total_gains = sum(t['profit_pct'] for t in winning_trades)
    total_losses = abs(sum(t['profit_pct'] for t in losing_trades))
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
    
    # MFE/MAE Analysis
    avg_mfe = np.mean([t['max_gain_pct'] for t in trades])
    avg_mae = np.mean([t['max_loss_pct'] for t in trades])
    mfe_mae_ratio = abs(avg_mfe / avg_mae) if avg_mae != 0 else float('inf')

    # Risk Metrics
    returns = np.array([t['profit_pct'] for t in trades])
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    # Sortino Ratio (downside risk only)
    downside_returns = returns[returns < 0]
    sortino_ratio = np.mean(returns) / np.std(downside_returns) if len(downside_returns) > 0 else 0

    # Drawdown Calculation
    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = (rolling_max - equity_series) / rolling_max * 100
    max_drawdown = drawdown.max()

    # Benchmark (Buy & Hold)
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    strategy_return = equity[-1] - 100  # Final equity - initial 100%
    outperformance = strategy_return - buy_hold_return
    
    # --- New Metrics ---
    # 1. Annualized Return Calculation
    first_date = df['Time'].iloc[0]
    last_date = df['Time'].iloc[-1]
    total_days = (last_date - first_date).days
    total_years = total_days / 365.25
    annualized_return_pct = ((equity[-1] / 100) ** (1/total_years) - 1) * 100 if total_years > 0 else 0
    
    # 2. Average Holding Days per Trade
    avg_holding_days = np.mean([t['holding_days'] for t in trades])

    # --- Step 3: Return Results ---
    results = {
        # Trade Summary
        "total_trades": int(total_trades),  # Convert to native int
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "hit_rate": float(hit_rate),
        
        # Profitability
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "max_win_pct": float(max_win),
        "max_loss_pct":float(max_loss),
        "win_loss_ratio": float(win_loss_ratio),
        "profit_factor": float(profit_factor),
        "expected_value_pct": float(expected_value),
        
        # Trade Management
        "avg_mfe_pct": float(avg_mfe),
        "avg_mae_pct": float(avg_mae),
        "mfe_mae_ratio": float(mfe_mae_ratio),
        "avg_holding_days": float(avg_holding_days),
        
        # Risk Metrics
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown_pct": float(max_drawdown),
        "annualized_return_pct": float(annualized_return_pct),
        
        # Benchmark
        "buy_hold_return_pct": float(buy_hold_return),
        "strategy_return_pct": float(strategy_return),
        "outperformance_pct": float(outperformance),
        
        # Raw Data
        "trades": trades[-2:],  # First 10 trades for inspection
        # "equity_curve": equity  # For plotting
    }
    
    # Optionally round all float values to 2 decimal places
    for key, value in results.items():
        if isinstance(value, float):
            results[key] = round(value, 2)
    
    return results

def get_market_data(symbol: str):
    """Get enhanced market data including sector and fundamental metrics"""
    try:
        # base_symbol = symbol.replace('.NS', '').replace('.BO', '')
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        market_cap = info.get('marketCap', 0)
        market_cap_cr = market_cap / 1e7 if market_cap else 0
        
        return {
            'market_cap_cr': market_cap_cr,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0)
        }
    except Exception:
        return {
            'market_cap_cr': 0,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'pe_ratio': 0,
            'pb_ratio': 0,
            'debt_to_equity': 0,
            'roe': 0
        }





def get_res(symbol):
    df, ticker_symbol = DataFetcher(symbol, interval='1d', min_years=5)
    if df.empty:
        return None
        
    df = calculate_supertrend(df)
    df = price_sp_wma(df)
    
    # Calculate data points
    data, labels = calculate_data_points(df)
    
    # Ensure we have matching lengths
    df = df.iloc[len(df) - len(data):].copy()
    
    # Calculate KNN values
    knn_results = []
    for i in range(len(df)):
        knn_value = knn_weighted_series(data[i], labels[i], 3, df['superTrend'].iloc[i])
        knn_results.append(knn_value)
    
    df['label_'] = knn_results
    df['longTrailingStop'] = df['superTrend'] - (df['atr'] * 3)
    df = apply_corrected_trading_logic(df)
    
    
    stock_info = get_market_data(ticker_symbol)
    
    df = df[['Time', 'close', 'signal']]
    
    back_test_met= back_test_metrice(df)
    df = df[df['signal'] != 'none']
    if not df.empty:
        return {
            'symbol': ticker_symbol,
            'Date': df['Time'].iloc[-1],
            'Current_price': df['close'].iloc[-1],
            'signal': df['signal'].iloc[-1],
            **back_test_met,
            **stock_info
        }
    return None,None
  
def convert_for_json(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, np.generic):
        return obj.item()
    return str(obj)

if __name__ == "__main__":
    start_time = time.time()
    res = get_res('CDSL')
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(json.dumps(res, indent=4, default=convert_for_json))