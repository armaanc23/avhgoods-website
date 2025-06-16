import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd # Added for better data handling with Alpha Vantage
import numpy as np # For np.log, np.sqrt, np.exp
from scipy.stats import norm # For N(x), the CDF of standard normal distribution
import datetime as dt # For date calculations
import streamlit as st # For UI
import matplotlib # Import matplotlib directly
matplotlib.use('Agg') # Set backend BEFORE pyplot is imported
import matplotlib.pyplot as plt # For plotting

# Import py_vollib for option pricing
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks import analytical # For delta, and can get d1, d2 from here if needed
from py_vollib.black_scholes.implied_volatility import implied_volatility # For later if needed

# Import from our new data_sources module
from data_sources.finnhub_client import get_stock_quote_finnhub, finnhub_client # also import client to check initialization
from data_sources.polygon_client import ( # Import all necessary functions
    polygon_rest_client, polygon_init_error, 
    get_option_expirations_polygon, get_options_chain_polygon,
    get_polygon_stock_aggregates, get_ticker_news_polygon,
    get_current_stock_price_polygon, calculate_historical_volatility_polygon,
    calculate_sma_polygon, calculate_ema_polygon, calculate_rsi_polygon # New TI functions
)

# --- Configuration & API Keys ---
# IMPORTANT: For production, move API keys to Streamlit secrets (st.secrets)
# Alpha Vantage Configuration
ALPHA_VANTAGE_API_KEY = "W4FTROR58IZ6YVYH" # User provided Alpha Vantage key
USE_ALPHA_VANTAGE_ACTUALLY = True # Set to False to disable Alpha Vantage calls

# Finnhub Configuration
# The Finnhub API key is managed within finnhub_client.py for now.
# We can check if the client was initialized to determine if Finnhub is usable.
FINNHUB_ENABLED = True if finnhub_client is not None else False

# Polygon.io Configuration
POLYGON_ENABLED = True if polygon_rest_client is not None else False
POLYGON_ERROR_MESSAGE = polygon_init_error # Store the error message if any

RISK_FREE_RATE = 0.02 # Default risk-free rate (e.g., 2%)
ANNUAL_DIVIDEND_YIELD = 0.0 # Assume no dividends for now

def black_scholes_pop(S, K, T, r, v, option_type='call', q=ANNUAL_DIVIDEND_YIELD):
    """
    Calculates option price, delta, gamma, theta, and vega for the seller 
    using py_vollib (Black-Scholes-Merton model).

    S: Current stock price
    K: Option strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate (annualized)
    v: Implied volatility (annualized)
    option_type: 'call' or 'put'
    q: Annual dividend yield (continuous compounding)

    Returns: A dictionary {'price': float, 'delta': float, 'gamma': float, 'theta': float, 'vega': float, 'pop_seller': float} 
             or None if T or v is zero/negative, or other input errors.
    POP for seller is the probability the option expires out-of-the-money.
    """
    flag = option_type[0].lower() # 'c' for call, 'p' for put

    if T <= 0: # Option expired or expires today
        price = 0.0
        delta = 0.0
        gamma = 0.0
        theta = 0.0
        vega = 0.0
        pop_seller = 0.0
        if option_type == 'call':
            price = max(0, S - K)
            delta = 1.0 if S > K else (0.5 if S == K else 0.0) # Simplified delta for T=0
            pop_seller = 1.0 if S <= K else 0.0 # Expires OTM if S <= K
        elif option_type == 'put':
            price = max(0, K - S)
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0) # Simplified delta for T=0
            pop_seller = 1.0 if S >= K else 0.0 # Expires OTM if S >= K
        return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'pop_seller': pop_seller}

    if v <= 0: # Volatility must be positive for py_vollib
        # print(f"Debug BS (py_vollib): v={v}. Cannot calculate with non-positive v.")
        return None # Or handle as in T=0 if appropriate, but py_vollib will error

    try:
        # Calculate Price using py_vollib
        bs_price = black_scholes(flag, S, K, T, r, v, q)
        
        # Calculate Delta using py_vollib
        bs_delta = analytical.delta(flag, S, K, T, r, v, q)
        # Calculate other Greeks using py_vollib
        bs_gamma = analytical.gamma(flag, S, K, T, r, v, q)
        bs_theta = analytical.theta(flag, S, K, T, r, v, q)
        bs_vega = analytical.vega(flag, S, K, T, r, v, q)

        # Calculate d1 and d2 for POP calculation (py_vollib doesn't directly return them from price/delta functions)
        # We can reconstruct d1 and d2 using the formulas if analytical.d1 and analytical.d2 are not available directly
        # d1 = (ln(S/K) + (r - q + 0.5 * v^2) * T) / (v * sqrt(T))
        # d2 = d1 - v * sqrt(T)
        # However, py_vollib.black_scholes.greeks.analytical often contains d1 and d2, let's check its structure or derive.
        # For simplicity, let's stick to the standard formulas for d1, d2 if not easily extracted.
        # Python's log is natural log (ln)
        
        # Corrected d1, d2 calculation incorporating dividend yield q
        d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
        
        pop_seller = 0.0
        if option_type == 'call':
            pop_seller = norm.cdf(-d2)  # Probability S_T <= K
        elif option_type == 'put':
            pop_seller = norm.cdf(d2)   # Probability S_T >= K
        else:
            return None # Should not happen if flag is 'c' or 'p'
            
        return {'price': bs_price, 'delta': bs_delta, 'pop_seller': pop_seller}

    except ZeroDivisionError:
        # This can happen if v * np.sqrt(T) is zero, which is checked by v<=0 and T<=0, but as a safeguard.
        # print(f"Debug BS (py_vollib): ZeroDivisionError for S={S}, K={K}, T={T}, r={r}, v={v}, type={option_type}")
        return None # Fallback to previous behavior of returning None for such cases
    except ValueError as e:
        # py_vollib can raise ValueError for invalid inputs (e.g. negative price)
        # print(f"Debug BS (py_vollib): ValueError for S={S}, K={K}, T={T}, r={r}, v={v}, type={option_type}. Error: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors from py_vollib
        # print(f"Debug BS (py_vollib): Unexpected error for S={S}, K={K}, T={T}, r={r}, v={v}, type={option_type}. Error: {e}")
        return None

def get_options_chain_yf(ticker_symbol):
    """
    Fetches the options chain for a given stock ticker using Yahoo Finance.
    """
    ticker = yf.Ticker(ticker_symbol)
    raw_expirations = ticker.options

    if not raw_expirations:
        st.warning(f"No options expirations found for {ticker_symbol} via yfinance")
        return None

    # Comprehensive cleaning function for date strings
    def clean_date_str(date_str):
        if not isinstance(date_str, str):
            return date_str # Or handle error, e.g., return a default or raise
        unicode_hyphens = [
            '\u2212',  # MINUS SIGN
            '\u2010',  # HYPHEN
            '\u2011',  # NON-BREAKING HYPHEN
            '\u2012',  # FIGURE DASH
            '\u2013',  # EN DASH
            '\u2014',  # EM DASH
            '\u2015',  # HORIZONTAL BAR
        ]
        cleaned_str = date_str
        for char_to_replace in unicode_hyphens:
            cleaned_str = cleaned_str.replace(char_to_replace, '-')
        return cleaned_str

    # Clean all expiration date strings upon fetching
    expirations = [clean_date_str(exp_date) for exp_date in raw_expirations]

    options_data = {}
    for original_expiry, cleaned_expiry in zip(raw_expirations, expirations):
        try:
            # Use original_expiry to fetch from yfinance as it expects that format
            options_chain = ticker.option_chain(original_expiry)
            # Store with cleaned_expiry as the key for consistent use later
            options_data[cleaned_expiry] = {
                "calls": options_chain.calls,
                "puts": options_chain.puts
            }
        except Exception as e:
            st.error(f"Could not fetch options for {ticker_symbol} on {original_expiry} (cleaned: {cleaned_expiry}) from yfinance: {e}")
            continue # Skip to next expiration if one fails
    return options_data

def get_historical_prices_yf(ticker_symbol, period="1y", interval="1d"):
    """
    Fetches historical stock prices for a given ticker using Yahoo Finance.
    Args:
        ticker_symbol (str): The stock ticker symbol.
        period (str): The period for which to fetch data (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
        interval (str): The interval of data points (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            st.warning(f"No historical data found for {ticker_symbol} for period {period} and interval {interval}.")
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching historical prices for {ticker_symbol} from yfinance: {e}")
        return None

def get_current_stock_price_av(ticker_symbol):
    """
    Fetches the latest closing price for a stock using Alpha Vantage.
    """
    if not USE_ALPHA_VANTAGE_ACTUALLY:
        st.sidebar.warning("Alpha Vantage calls are currently disabled (API key might be placeholder).")
        return None
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY" or ALPHA_VANTAGE_API_KEY == "W4FTROR58IZ6YVYH" and not USE_ALPHA_VANTAGE_ACTUALLY:
        st.sidebar.error("Alpha Vantage API key not properly set or is a placeholder.")
        return None
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ts.get_quote_endpoint(symbol=ticker_symbol)
        if not data.empty and '05. price' in data.columns:
            return float(data['05. price'].iloc[0])
        else:
            # st.warning(f"Could not retrieve current price for {ticker_symbol} from Alpha Vantage. Data: {data}")
            return None
    except Exception as e:
        st.error(f"Error fetching current price for {ticker_symbol} from Alpha Vantage: {e}")
        return None

def get_sma_av(ticker_symbol, interval='daily', time_period=20, series_type='close'):
    """
    Fetches the Simple Moving Average (SMA) for a stock using Alpha Vantage.
    """
    if not USE_ALPHA_VANTAGE_ACTUALLY: return None
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY" or ALPHA_VANTAGE_API_KEY == "W4FTROR58IZ6YVYH" and not USE_ALPHA_VANTAGE_ACTUALLY: return None
    ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ti.get_sma(symbol=ticker_symbol, interval=interval,
                                     time_period=time_period, series_type=series_type)
        if not data.empty:
            # The data is returned with the most recent first.
            return data # Returns a DataFrame with 'SMA' column
        else:
            print(f"Could not retrieve SMA for {ticker_symbol} from Alpha Vantage. Data: {data}")
            return None
    except Exception as e:
        print(f"Error fetching SMA for {ticker_symbol} from Alpha Vantage: {e}")
        return None

def get_ema_av(ticker_symbol, interval='daily', time_period=20, series_type='close'):
    """
    Fetches the Exponential Moving Average (EMA) for a stock using Alpha Vantage.
    """
    if not USE_ALPHA_VANTAGE_ACTUALLY: return None
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY" or ALPHA_VANTAGE_API_KEY == "W4FTROR58IZ6YVYH" and not USE_ALPHA_VANTAGE_ACTUALLY: return None
    ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ti.get_ema(symbol=ticker_symbol, interval=interval,
                                     time_period=time_period, series_type=series_type)
        if not data.empty:
            return data # Returns a DataFrame with 'EMA' column
        else:
            print(f"Could not retrieve EMA for {ticker_symbol} from Alpha Vantage. Data: {data}")
            return None
    except Exception as e:
        print(f"Error fetching EMA for {ticker_symbol} from Alpha Vantage: {e}")
        return None

def get_macd_av(ticker_symbol, interval='daily', series_type='close', fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Fetches the Moving Average Convergence Divergence (MACD) for a stock using Alpha Vantage.
    """
    if not USE_ALPHA_VANTAGE_ACTUALLY: return None
    # This endpoint is often premium, so expect it to fail on free keys
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY" or ALPHA_VANTAGE_API_KEY == "W4FTROR58IZ6YVYH" and not USE_ALPHA_VANTAGE_ACTUALLY: return None
    ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ti.get_macd(symbol=ticker_symbol, interval=interval, series_type=series_type,
                                      fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        if not data.empty:
            return data # Returns a DataFrame with 'MACD', 'MACD_Signal', 'MACD_Hist' columns
        else:
            print(f"Could not retrieve MACD for {ticker_symbol} from Alpha Vantage. Data: {data}")
            return None
    except Exception as e:
        print(f"Error fetching MACD for {ticker_symbol} from Alpha Vantage: {e}")
        return None

def get_rsi_av(ticker_symbol, interval='daily', time_period=14, series_type='close'):
    """
    Fetches the Relative Strength Index (RSI) for a stock using Alpha Vantage.
    """
    if not USE_ALPHA_VANTAGE_ACTUALLY: return None
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY" or ALPHA_VANTAGE_API_KEY == "W4FTROR58IZ6YVYH" and not USE_ALPHA_VANTAGE_ACTUALLY: return None
    ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ti.get_rsi(symbol=ticker_symbol, interval=interval,
                                     time_period=time_period, series_type=series_type)
        if not data.empty:
            return data # Returns a DataFrame with 'RSI' column
        else:
            print(f"Could not retrieve RSI for {ticker_symbol} from Alpha Vantage. Data: {data}")
            return None
    except Exception as e:
        print(f"Error fetching RSI for {ticker_symbol} from Alpha Vantage: {e}")
        return None

def calculate_macd_pandas(historical_prices_df, fast_period=12, slow_period=26, signal_period=9, price_column='Close'):
    """
    Calculates MACD, Signal Line, and MACD Histogram from historical price data using pandas.

    Args:
        historical_prices_df (pd.DataFrame): DataFrame with historical prices, must contain 'price_column'.
        fast_period (int): The time period for the fast EMA.
        slow_period (int): The time period for the slow EMA.
        signal_period (int): The time period for the signal line EMA.
        price_column (str): The name of the column containing the price data (e.g., 'Close').

    Returns:
        pd.DataFrame: Original DataFrame with added 'EMA_fast', 'EMA_slow', 'MACD', 'Signal_Line', 'MACD_Histogram' columns.
                     Returns None if the price_column is not found or data is insufficient.
    """
    if price_column not in historical_prices_df.columns:
        print(f"Price column '{price_column}' not found in historical data for MACD calculation.")
        return None

    if len(historical_prices_df) < slow_period: # Need enough data for the slowest EMA
        print(f"Insufficient data for MACD calculation. Need at least {slow_period} periods, got {len(historical_prices_df)}.")
        return None

    df = historical_prices_df.copy()
    df['EMA_fast'] = df[price_column].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df[price_column].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

def analyze_options_data(options_df, current_stock_price, expiration_date_str, option_type_str, option_chain_source_str="yfinance", min_volume=10, min_open_interest=10, moneyness_strikes_range=5, risk_free_rate=RISK_FREE_RATE):
    """
    Analyzes and filters options data for a single expiration date, calculates BS delta and POP.
    Args:
        options_df (pd.DataFrame): DataFrame of options (calls or puts) for one expiry.
        current_stock_price (float): Current price of the underlying stock.
        expiration_date_str (str): The expiration date as a string (e.g., "YYYY-MM-DD").
        option_type_str (str): 'calls' or 'puts' (for messages and determining BS type).
        option_chain_source_str (str): Source of the options data (e.g., "Polygon.io", "yfinance").
    """
    if options_df is None or options_df.empty or current_stock_price is None or expiration_date_str is None:
        print(f"analyze_options_data: Input validation failed. options_df empty: {options_df is None or options_df.empty}, current_stock_price: {current_stock_price}, expiry: {expiration_date_str}")
        return None, pd.NA

    # Define expected columns based on source. Polygon provides greeks directly.
    # yfinance might provide some, but we recalculate for consistency using py_vollib via black_scholes_pop.
    # Common required columns for initial filtering regardless of source:
    required_cols = ['strike', 'volume', 'openInterest', 'impliedVolatility', 'lastPrice'] 
    if option_chain_source_str == "Polygon.io":
        # Polygon source should have these pre-populated and correctly named from get_options_chain_polygon
        required_cols.extend(['delta', 'gamma', 'theta', 'vega']) # These are expected to be present
        # We will rename them to bs_delta etc. later for consistency if they exist.

    if not all(col in options_df.columns for col in required_cols if col not in ['delta', 'gamma', 'theta', 'vega'] or option_chain_source_str != "Polygon.io"):
        # Check only essential non-Greek cols if Polygon source, as Greeks are handled differently
        essential_cols = ['strike', 'volume', 'openInterest', 'impliedVolatility', 'lastPrice']
        missing_essentials = [col for col in essential_cols if col not in options_df.columns]
        if missing_essentials:
            print(f"analyze_options_data: Missing essential columns: {missing_essentials} in {option_type_str} df from {option_chain_source_str}. Columns: {options_df.columns}")
            return None, pd.NA
    
    filtered_options = options_df.copy()
    # Ensure numeric types for calculation columns, coercing errors and filling NaNs for volume/OI
    for col in ['volume', 'openInterest']:
        filtered_options[col] = pd.to_numeric(filtered_options[col], errors='coerce').fillna(0)
    for col in ['strike', 'impliedVolatility', 'lastPrice', 'delta', 'gamma', 'theta', 'vega']:
        if col in filtered_options.columns:
            filtered_options[col] = pd.to_numeric(filtered_options[col], errors='coerce')

    # Drop rows if essential data for Black-Scholes (or direct use) is missing AFTER coercion
    # IV must be positive for BS calc. If using Polygon IV, it should also be positive.
    filtered_options.dropna(subset=['strike', 'lastPrice'], inplace=True) 
    filtered_options = filtered_options[filtered_options['impliedVolatility'] > 0]

    filtered_options = filtered_options[
        (filtered_options['volume'] >= min_volume) &
        (filtered_options['openInterest'] >= min_open_interest) &
        (filtered_options['impliedVolatility'] > 0) # Volatility must be positive for BS
    ]

    if filtered_options.empty:
        return None, pd.NA

    # Calculate Time to Expiration (T)
    # Assuming yfinance option chain expiration_date_str is like 'YYYY-MM-DD'
    try:
        expiry_dt = dt.datetime.strptime(expiration_date_str, '%Y-%m-%d')
        # yfinance option_chain gives expiry for market open, so add hours to represent end of expiry day for T calculation.
        # For simplicity, let's consider T to be (expiry_date - today).days / 365. More precise would be trading days or account for exact time.
        # For POP, usually use (days to last trading day + 1) / 365
        # Let's use (expiry_dt - today).days. Ensure expiry_dt is in the future.
        # Add 1 to include the expiration day itself for a more standard DTE calculation from today.
        time_to_expiry_days = (expiry_dt - dt.datetime.now()).days + 1 
        if time_to_expiry_days <= 0:
            T = 0 # Option expired or expires today
        else:
            T = time_to_expiry_days / 365.25
    except ValueError:
        # print(f"Error parsing expiration date: {expiration_date_str}")
        return None, pd.NA

    # Apply Black-Scholes if T > 0
    bs_results = []
    if T > 0:
        for index, row in filtered_options.iterrows():
            S = current_stock_price
            K = row['strike']
            v = row['impliedVolatility']
            # Determine option type for Black-Scholes ('call' or 'put')
            bs_option_type = 'call' if option_type_str == 'calls' else 'put' 
            
            bs_val = black_scholes_pop(S, K, T, risk_free_rate, v, option_type=bs_option_type)
            if bs_val:
                bs_results.append({
                    'strike': K,
                    'bs_delta': bs_val['delta'], 
                    'POP_bs': bs_val['pop_seller'], 
                    'bs_price': bs_val['price']
                })
            else:
                bs_results.append({'strike': K, 'bs_delta': pd.NA, 'POP_bs': pd.NA, 'bs_price': pd.NA})
        
        if bs_results:
            bs_df = pd.DataFrame(bs_results)
            filtered_options = pd.merge(filtered_options, bs_df, on='strike', how='left')
        else:
            for col in ['bs_delta', 'POP_bs', 'bs_price']: filtered_options[col] = pd.NA
    else: # T <= 0, use simplified logic from black_scholes_pop for delta/pop if needed, or just mark as NA
        # For simplicity, if T=0, BS model as defined will handle it.
        # Or, we can pre-calculate and assign.
        # For now, let's assume black_scholes_pop handles T=0 for expired state.
        # If T is calculated as 0, BS function will return deterministic values.
        # Re-iterate for T=0 using the BS function logic for consistency
        for index, row in filtered_options.iterrows():
            S = current_stock_price
            K = row['strike']
            v = 0.01 # Dummy small vol for T=0 case if BS requires positive v
            bs_option_type = 'call' if option_type_str == 'calls' else 'put' 
            bs_val = black_scholes_pop(S, K, 0, risk_free_rate, v, option_type=bs_option_type) # Explicitly T=0
            bs_results.append({
                    'strike': K,
                    'bs_delta': bs_val['delta'], 
                    'POP_bs': bs_val['pop_seller'], 
                    'bs_price': bs_val['price']
                })
        if bs_results:
            bs_df = pd.DataFrame(bs_results)
            filtered_options = pd.merge(filtered_options, bs_df, on='strike', how='left')
        else: # Should not happen if filtered_options is not empty
            for col in ['bs_delta', 'POP_bs', 'bs_price']: filtered_options[col] = pd.NA

    # Moneyness and strike range filtering (moved after BS so all options get BS values if possible)
    if filtered_options.empty: # check after potential drops from BS merge issues or T=0 handling
        return None, pd.NA

    filtered_options.loc[:, 'moneyness_abs_diff'] = (filtered_options['strike'] - current_stock_price).abs()
    if filtered_options.empty:
        return None, pd.NA
    try:
        atm_strike_row = filtered_options.loc[filtered_options['moneyness_abs_diff'].idxmin()]
        atm_strike = atm_strike_row['strike']
    except ValueError: 
        return None, pd.NA 
    
    sorted_strikes = filtered_options['strike'].sort_values().unique()
    if atm_strike not in sorted_strikes:
        if len(sorted_strikes) == 0: return None, pd.NA
        atm_strike = sorted_strikes[pd.Series(sorted_strikes).sub(current_stock_price).abs().idxmin()]

    atm_idx_in_sorted_unique_strikes = pd.Series(sorted_strikes).searchsorted(atm_strike)
    lower_bound_idx = max(0, atm_idx_in_sorted_unique_strikes - moneyness_strikes_range)
    upper_bound_idx = min(len(sorted_strikes), atm_idx_in_sorted_unique_strikes + moneyness_strikes_range + 1)
    selected_strikes = sorted_strikes[lower_bound_idx:upper_bound_idx]
    final_filtered_options = filtered_options[filtered_options['strike'].isin(selected_strikes)].copy()

    if final_filtered_options.empty:
        return None, pd.NA
        
    final_filtered_options.loc[:, 'distance_from_spot'] = final_filtered_options['strike'] - current_stock_price
    
    # Remove old POP_delta as we are replacing it with POP_bs
    if 'POP_delta' in final_filtered_options.columns:
        final_filtered_options = final_filtered_options.drop(columns=['POP_delta'])
    if 'delta' in final_filtered_options.columns and 'bs_delta' not in final_filtered_options.columns: # if yf delta existed but bs_delta failed
        final_filtered_options = final_filtered_options.drop(columns=['delta'])
    elif 'delta' in final_filtered_options.columns and 'bs_delta' in final_filtered_options.columns: # if both exist, prefer bs_delta and remove original yf delta
         final_filtered_options = final_filtered_options.drop(columns=['delta'])

    display_cols = ['strike', 'lastPrice', 'bs_price', 'volume', 'openInterest', 'impliedVolatility', 'bs_delta', 'POP_bs', 'distance_from_spot']
    display_cols_present = [col for col in display_cols if col in final_filtered_options.columns]
    
    avg_atm_iv = pd.NA
    if 'impliedVolatility' in final_filtered_options.columns and not final_filtered_options['impliedVolatility'].empty:
        avg_atm_iv = final_filtered_options['impliedVolatility'].mean()
        
    return final_filtered_options[display_cols_present].sort_values(by='strike'), avg_atm_iv

# --- Helper function to find options ---
def find_option_by_criteria(df, option_type, strategy_type, current_price, target_delta=None, strikes_away=None, preferred_side='otm'):
    """
    Finds a suitable option contract from a filtered DataFrame.
    df: DataFrame of filtered options (calls or puts).
    option_type: 'call' or 'put'.
    strategy_type: Helps determine preference e.g. 'long' (buy), 'short' (sell).
    current_price: Current stock price.
    target_delta: Approximate delta to look for (e.g., 0.5 for ATM, 0.3 for OTM sell).
    strikes_away: Number of strikes OTM.
    preferred_side: 'otm', 'itm', 'atm'.
    """
    if df is None or df.empty:
        return None

    temp_df = df.copy()
    temp_df['abs_delta_diff'] = pd.NA
    temp_df['strikes_from_atm'] = pd.NA

    # Find ATM strike first
    atm_strike_val = temp_df.iloc[(temp_df['strike'] - current_price).abs().argsort()[:1]]['strike'].values[0]

    if target_delta and 'bs_delta' in temp_df.columns:
        temp_df['abs_delta_diff'] = (temp_df['bs_delta'].abs() - target_delta).abs()
        selected_option = temp_df.sort_values(by='abs_delta_diff').iloc[0]
        return selected_option
    
    if strikes_away is not None:
        if option_type == 'call':
            if preferred_side == 'otm':
                potential_options = temp_df[temp_df['strike'] > atm_strike_val].sort_values(by='strike')
            else: # itm
                potential_options = temp_df[temp_df['strike'] < atm_strike_val].sort_values(by='strike', ascending=False)
        else: # put
            if preferred_side == 'otm':
                potential_options = temp_df[temp_df['strike'] < atm_strike_val].sort_values(by='strike', ascending=False)
            else: # itm
                potential_options = temp_df[temp_df['strike'] > atm_strike_val].sort_values(by='strike')
        
        if not potential_options.empty and len(potential_options) >= strikes_away:
            return potential_options.iloc[strikes_away -1] # 1-indexed strikes_away
        elif not potential_options.empty: # Not enough, return furthest/closest OTM/ITM
            return potential_options.iloc[-1 if preferred_side == 'otm' else 0] 

    # Fallback to ATM if other criteria don't yield result or not specified
    return temp_df.iloc[(temp_df['strike'] - current_price).abs().argsort()[:1]].iloc[0]

def recommend_strategies(indicators, analyzed_calls_df, analyzed_puts_df, current_stock_price, selected_expiry_str, ticker_symbol):
    general_recommendations = [] # For text like bias, IV
    specific_plays_list_of_dicts = [] # For table data - list of dictionaries

    price = indicators.get('current_price'); sma20 = indicators.get('final_sma20'); sma50 = indicators.get('final_sma50')
    rsi = indicators.get('rsi_14_av'); macd_info = indicators.get('macd_pandas'); avg_iv = indicators.get('avg_atm_iv')
    bias = "Neutral"; bullish_signals = 0; bearish_signals = 0

    if price and sma20 and price > sma20: bullish_signals += 1
    if price and sma20 and price < sma20: bearish_signals += 1
    if price and sma50 and price > sma50: bullish_signals += 1
    if price and sma50 and price < sma50: bearish_signals += 1
    if macd_info:
        if macd_info['MACD'] > macd_info['Signal_Line'] and macd_info['MACD_Histogram'] > 0.05 : bullish_signals +=1
        if macd_info['MACD'] < macd_info['Signal_Line'] and macd_info['MACD_Histogram'] < -0.05: bearish_signals +=1
    if rsi:
        if rsi < 30: bullish_signals += 1
        if rsi > 70: bearish_signals += 1
    
    if bullish_signals > bearish_signals + 1: bias = "Bullish"
    elif bearish_signals > bullish_signals + 1: bias = "Bearish"
    
    general_recommendations.append(f"**Overall Technical Bias: {bias}** (Bullish: {bullish_signals}, Bearish: {bearish_signals}) for {ticker_symbol})")
    
    iv_level = "Moderate"; iv_text = "Not Available"
    if avg_iv is not None and not pd.isna(avg_iv):
        if avg_iv > 0.40: iv_level = "High"
        elif avg_iv < 0.20: iv_level = "Low"
        iv_text = f"{avg_iv:.4f} (Considered: {iv_level})"
    general_recommendations.append(f"**Implied Volatility (ATM avg for {selected_expiry_str}):** {iv_text}")
    
    cleaned_expiry_str = str(selected_expiry_str) 
    unicode_hyphens = ['\\u2212','\\u2010','\\u2011','\\u2012','\\u2013','\\u2014','\\u2015']
    for char in unicode_hyphens:
        cleaned_expiry_str = cleaned_expiry_str.replace(char, '-')

    calls_available = analyzed_calls_df is not None and not analyzed_calls_df.empty
    puts_available = analyzed_puts_df is not None and not analyzed_puts_df.empty
    aggressive_otm_strikes = 2 
    spread_width_index = 1 

    if bias == "Bullish":
        general_recommendations.append(f"General Bullish Idea for {cleaned_expiry_str}: Consider Long Calls, Bull Call Spreads, or Bull Put Spreads.")
        if calls_available and len(specific_plays_list_of_dicts) < 3:
            lc_candidate = find_option_by_criteria(analyzed_calls_df, 'call', 'long', current_stock_price, strikes_away=aggressive_otm_strikes, preferred_side='otm')
            if lc_candidate is None: lc_candidate = find_option_by_criteria(analyzed_calls_df, 'call', 'long', current_stock_price, strikes_away=1, preferred_side='otm')
            if lc_candidate is None: lc_candidate = find_option_by_criteria(analyzed_calls_df, 'call', 'long', current_stock_price)
            if lc_candidate is not None:
                delta = lc_candidate.get('bs_delta', pd.NA)
                specific_plays_list_of_dicts.append({
                    "Strategy": "Long Call", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                    "Leg 1": f"Buy Call @ {lc_candidate['strike']:.2f}", "Leg 2": "",
                    "Premium": f"${lc_candidate['lastPrice']:.2f} Debit", 
                    "Metric": f"Delta: {delta:.2f}" if pd.notna(delta) else "Delta: N/A",
                    "Rationale": "Directional bet on price increase (more OTM for higher potential profit/risk)."
                })
        if calls_available and len(analyzed_calls_df) >= (spread_width_index + 2) and len(specific_plays_list_of_dicts) < 3:
            bcs_long_leg = find_option_by_criteria(analyzed_calls_df, 'call', 'long', current_stock_price, strikes_away=0, preferred_side='atm')
            if bcs_long_leg is None: bcs_long_leg = find_option_by_criteria(analyzed_calls_df, 'call', 'long', current_stock_price, strikes_away=1, preferred_side='otm')
            if bcs_long_leg is not None:
                bcs_potential_short_legs = analyzed_calls_df[analyzed_calls_df['strike'] > bcs_long_leg['strike']].sort_values(by='strike')
                if len(bcs_potential_short_legs) > spread_width_index:
                    bcs_short_leg = bcs_potential_short_legs.iloc[spread_width_index]
                    net_debit = bcs_long_leg['lastPrice'] - bcs_short_leg['lastPrice']
                    if net_debit > 0:
                        specific_plays_list_of_dicts.append({
                            "Strategy": "Bull Call Spread", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                            "Leg 1": f"Buy Call @ {bcs_long_leg['strike']:.2f}", 
                            "Leg 2": f"Sell Call @ {bcs_short_leg['strike']:.2f}",
                            "Premium": f"${net_debit:.2f} Debit", "Metric": "Defined Risk/Reward",
                            "Rationale": "Limits cost for moderate price increase (wider for higher potential profit/risk)."
                        })
        if puts_available and len(analyzed_puts_df) >= (spread_width_index + 2) and len(specific_plays_list_of_dicts) < 3:
            bps_short_leg = find_option_by_criteria(analyzed_puts_df, 'put', 'short', current_stock_price, strikes_away=aggressive_otm_strikes, preferred_side='otm')
            if bps_short_leg is None: bps_short_leg = find_option_by_criteria(analyzed_puts_df, 'put', 'short', current_stock_price, strikes_away=1, preferred_side='otm')
            if bps_short_leg is not None:
                bps_potential_long_legs = analyzed_puts_df[analyzed_puts_df['strike'] < bps_short_leg['strike']].sort_values(by='strike', ascending=False)
                if len(bps_potential_long_legs) > spread_width_index:
                    bps_long_leg = bps_potential_long_legs.iloc[spread_width_index]
                    net_credit = bps_short_leg['lastPrice'] - bps_long_leg['lastPrice']
                    if net_credit > 0:
                        pop = bps_short_leg.get('POP_bs', pd.NA)
                        specific_plays_list_of_dicts.append({
                            "Strategy": "Bull Put Spread", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                            "Leg 1": f"Sell Put @ {bps_short_leg['strike']:.2f}", 
                            "Leg 2": f"Buy Put @ {bps_long_leg['strike']:.2f}",
                            "Premium": f"${net_credit:.2f} Credit", 
                            "Metric": f"POP (Short Leg): {pop:.2f}" if pd.notna(pop) else "POP: N/A",
                            "Rationale": "Collect premium if stock stays above short strike; defined risk."
                        })
    elif bias == "Bearish":
        general_recommendations.append(f"General Bearish Idea for {cleaned_expiry_str}: Consider Long Puts, Bear Put Spreads, or Bear Call Spreads.")
        if puts_available and len(specific_plays_list_of_dicts) < 3:
            lp_candidate = find_option_by_criteria(analyzed_puts_df, 'put', 'long', current_stock_price, strikes_away=aggressive_otm_strikes, preferred_side='otm')
            if lp_candidate is None: lp_candidate = find_option_by_criteria(analyzed_puts_df, 'put', 'long', current_stock_price, strikes_away=1, preferred_side='otm')
            if lp_candidate is None: lp_candidate = find_option_by_criteria(analyzed_puts_df, 'put', 'long', current_stock_price)
            if lp_candidate is not None:
                delta = lp_candidate.get('bs_delta', pd.NA)
                specific_plays_list_of_dicts.append({
                    "Strategy": "Long Put", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                    "Leg 1": f"Buy Put @ {lp_candidate['strike']:.2f}", "Leg 2": "",
                    "Premium": f"${lp_candidate['lastPrice']:.2f} Debit", 
                    "Metric": f"Delta: {delta:.2f}" if pd.notna(delta) else "Delta: N/A",
                    "Rationale": "Directional bet on price decrease (more OTM for higher potential profit/risk)."
                })
        if puts_available and len(analyzed_puts_df) >= (spread_width_index + 2) and len(specific_plays_list_of_dicts) < 3:
            bps_long_leg = find_option_by_criteria(analyzed_puts_df, 'put', 'long', current_stock_price, strikes_away=0, preferred_side='atm')
            if bps_long_leg is None: bps_long_leg = find_option_by_criteria(analyzed_puts_df, 'put', 'long', current_stock_price, strikes_away=1, preferred_side='otm')
            if bps_long_leg is not None:
                bps_potential_short_legs = analyzed_puts_df[analyzed_puts_df['strike'] < bps_long_leg['strike']].sort_values(by='strike', ascending=False)
                if len(bps_potential_short_legs) > spread_width_index:
                    bps_short_leg = bps_potential_short_legs.iloc[spread_width_index]
                    net_debit = bps_long_leg['lastPrice'] - bps_short_leg['lastPrice']
                    if net_debit > 0:
                        specific_plays_list_of_dicts.append({
                            "Strategy": "Bear Put Spread", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                            "Leg 1": f"Buy Put @ {bps_long_leg['strike']:.2f}", 
                            "Leg 2": f"Sell Put @ {bps_short_leg['strike']:.2f}",
                            "Premium": f"${net_debit:.2f} Debit", "Metric": "Defined Risk/Reward",
                            "Rationale": "Limits cost for moderate price decrease (wider for higher potential profit/risk)."
                        })
        if calls_available and len(analyzed_calls_df) >= (spread_width_index + 2) and len(specific_plays_list_of_dicts) < 3:
            bcs_short_leg = find_option_by_criteria(analyzed_calls_df, 'call', 'short', current_stock_price, strikes_away=aggressive_otm_strikes, preferred_side='otm')
            if bcs_short_leg is None: bcs_short_leg = find_option_by_criteria(analyzed_calls_df, 'call', 'short', current_stock_price, strikes_away=1, preferred_side='otm')
            if bcs_short_leg is not None:
                bcs_potential_long_legs = analyzed_calls_df[analyzed_calls_df['strike'] > bcs_short_leg['strike']].sort_values(by='strike')
                if len(bcs_potential_long_legs) > spread_width_index:
                    bcs_long_leg = bcs_potential_long_legs.iloc[spread_width_index]
                    net_credit = bcs_short_leg['lastPrice'] - bcs_long_leg['lastPrice']
                    if net_credit > 0:
                        pop = bcs_short_leg.get('POP_bs', pd.NA)
                        specific_plays_list_of_dicts.append({
                            "Strategy": "Bear Call Spread", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                            "Leg 1": f"Sell Call @ {bcs_short_leg['strike']:.2f}", 
                            "Leg 2": f"Buy Call @ {bcs_long_leg['strike']:.2f}",
                            "Premium": f"${net_credit:.2f} Credit", 
                            "Metric": f"POP (Short Leg): {pop:.2f}" if pd.notna(pop) else "POP: N/A",
                            "Rationale": "Collect premium if stock stays below short strike; defined risk."
                        })
    elif bias == "Neutral":
        if iv_level == "High":
            general_recommendations.append(f"General Neutral Idea (High IV for {cleaned_expiry_str}): Consider Short Puts, Short Calls, or Iron Condors.")
            if puts_available and len(specific_plays_list_of_dicts) < 3:
                sp_candidate = find_option_by_criteria(analyzed_puts_df, 'put', 'short', current_stock_price, target_delta=0.30, preferred_side='otm')
                if sp_candidate is None: sp_candidate = find_option_by_criteria(analyzed_puts_df, 'put', 'short', current_stock_price, strikes_away=2, preferred_side='otm')
                if sp_candidate is not None:
                    pop = sp_candidate.get('POP_bs', pd.NA)
                    specific_plays_list_of_dicts.append({
                        "Strategy": "Short Put", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                        "Leg 1": f"Sell Put @ {sp_candidate['strike']:.2f}", "Leg 2": "",
                        "Premium": f"${sp_candidate['lastPrice']:.2f} Credit", 
                        "Metric": f"POP Seller: {pop:.2f}" if pd.notna(pop) else "POP: N/A",
                        "Rationale": "Income if stock stays above strike; willing to buy stock if assigned."
                    })
            if calls_available and len(specific_plays_list_of_dicts) < 3:
                sc_candidate = find_option_by_criteria(analyzed_calls_df, 'call', 'short', current_stock_price, target_delta=0.30, preferred_side='otm')
                if sc_candidate is None: sc_candidate = find_option_by_criteria(analyzed_calls_df, 'call', 'short', current_stock_price, strikes_away=2, preferred_side='otm')
                if sc_candidate is not None:
                    pop = sc_candidate.get('POP_bs', pd.NA)
                    specific_plays_list_of_dicts.append({
                        "Strategy": "Short Call", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                        "Leg 1": f"Sell Call @ {sc_candidate['strike']:.2f}", "Leg 2": "",
                        "Premium": f"${sc_candidate['lastPrice']:.2f} Credit", 
                        "Metric": f"POP Seller: {pop:.2f}" if pd.notna(pop) else "POP: N/A",
                        "Rationale": "Income if stock stays below strike; willing to short stock if assigned."
                    })
            if calls_available and puts_available and len(analyzed_calls_df) >= (spread_width_index + 2) and \
               len(analyzed_puts_df) >= (spread_width_index + 2) and len(specific_plays_list_of_dicts) < 3:
                ic_short_put = find_option_by_criteria(analyzed_puts_df, 'put', 'short', current_stock_price, strikes_away=aggressive_otm_strikes, preferred_side='otm')
                ic_long_put, ic_short_call, ic_long_call = None, None, None
                if ic_short_put:
                    put_wings = analyzed_puts_df[analyzed_puts_df['strike'] < ic_short_put['strike']].sort_values(by='strike', ascending=False)
                    if len(put_wings) > spread_width_index: ic_long_put = put_wings.iloc[spread_width_index]
                ic_short_call = find_option_by_criteria(analyzed_calls_df, 'call', 'short', current_stock_price, strikes_away=aggressive_otm_strikes, preferred_side='otm')
                if ic_short_call:
                    call_wings = analyzed_calls_df[analyzed_calls_df['strike'] > ic_short_call['strike']].sort_values(by='strike')
                    if len(call_wings) > spread_width_index: ic_long_call = call_wings.iloc[spread_width_index]
                if ic_short_put and ic_long_put and ic_short_call and ic_long_call and \
                   ic_long_put['strike'] < ic_short_put['strike'] < current_stock_price < ic_short_call['strike'] < ic_long_call['strike']:
                    net_credit = (ic_short_put['lastPrice'] + ic_short_call['lastPrice']) - (ic_long_put['lastPrice'] + ic_long_call['lastPrice'])
                    if net_credit > 0:
                        specific_plays_list_of_dicts.append({
                            "Strategy": "Iron Condor", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                            "Leg 1": f"Sell Put @ {ic_short_put['strike']:.2f} & Sell Call @ {ic_short_call['strike']:.2f}", 
                            "Leg 2": f"Buy Put @ {ic_long_put['strike']:.2f} & Buy Call @ {ic_long_call['strike']:.2f} (Wings)",
                            "Premium": f"${net_credit:.2f} Credit", 
                            "Metric": "Defined Risk/Reward",
                            "Rationale": "Income in high IV, expects price in range. Wider wings for higher credit."
                        })
        else: 
            general_recommendations.append(f"General Neutral Idea (Low/Mod IV for {cleaned_expiry_str}): Consider Long Straddles/Strangles.")
            if calls_available and puts_available and len(specific_plays_list_of_dicts) < 3:
                atm_call = find_option_by_criteria(analyzed_calls_df, 'call', 'long', current_stock_price)
                atm_put = find_option_by_criteria(analyzed_puts_df, 'put', 'long', current_stock_price)
                if atm_call is not None and atm_put is not None:
                    cost = atm_call['lastPrice'] + atm_put['lastPrice']
                    if atm_call['strike'] == atm_put['strike']: 
                        if len(specific_plays_list_of_dicts) < 3:
                            specific_plays_list_of_dicts.append({
                                "Strategy": "Long Straddle", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                                "Leg 1": f"Buy Call @ {atm_call['strike']:.2f}", 
                                "Leg 2": f"Buy Put @ {atm_put['strike']:.2f}",
                                "Premium": f"${cost:.2f} Debit", "Metric": "Expect Large Move",
                                "Rationale": "Profits from large price move in either direction."
                            })
                    else: 
                         if len(specific_plays_list_of_dicts) < 3:
                            specific_plays_list_of_dicts.append({
                                "Strategy": "Long Strangle", "Ticker": ticker_symbol, "Expiry": cleaned_expiry_str,
                                "Leg 1": f"Buy Call @ {atm_call['strike']:.2f}", 
                                "Leg 2": f"Buy Put @ {atm_put['strike']:.2f}",
                                "Premium": f"${cost:.2f} Debit", "Metric": "Expect Large Move",
                                "Rationale": "Profits from large price move (diff. strikes)."
                            })
    if not specific_plays_list_of_dicts:
        general_recommendations.append("No specific option plays could be reliably constructed from the filtered options for the selected expiry based on the current rules.")
    return general_recommendations, specific_plays_list_of_dicts

def get_stock_info_yf(ticker_symbol):
    """
    Fetches general stock information including current price using Yahoo Finance.
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    # yfinance often provides 'regularMarketPrice', 'currentPrice', or 'previousClose'
    # We'll try to find a suitable price field.
    price_keys = ['regularMarketPrice', 'currentPrice', 'bid', 'ask', 'previousClose']
    current_price = None
    for key in price_keys:
        if key in info and info[key] is not None:
            current_price = info[key]
            break
    
    if current_price is None:
        st.warning(f"Could not determine current price for {ticker_symbol} from yfinance info.")

    return {"info": info, "current_price": current_price}

def plot_payoff(strategy_details, stock_price_at_expiration):
    """
    Calculates the P/L for a given strategy at a specific stock price at expiration.
    This is a generic P/L calculator that will be expanded for different strategies.
    
    Args:
        strategy_details (dict): Contains info about the strategy.
                                 For 'long_call': {'type': 'long_call', 'strike': float, 'premium': float}
                                 For 'short_call': {'type': 'short_call', 'strike': float, 'premium': float}
                                 For 'long_put': {'type': 'long_put', 'strike': float, 'premium': float}
                                 For 'short_put': {'type': 'short_put', 'strike': float, 'premium': float}
                                 Can be extended for spreads.
        stock_price_at_expiration (float): The stock price at expiration.

    Returns:
        float: Profit or Loss.
    """
    strat_type = strategy_details.get('type')
    strike = strategy_details.get('strike')
    premium = strategy_details.get('premium')
    
    if strat_type == 'long_call':
        return max(0, stock_price_at_expiration - strike) - premium
    elif strat_type == 'short_call':
        return premium - max(0, stock_price_at_expiration - strike)
    elif strat_type == 'long_put':
        return max(0, strike - stock_price_at_expiration) - premium
    elif strat_type == 'short_put':
        return premium - max(0, strike - stock_price_at_expiration)
    # Add more strategies (spreads) here later
    # Example: Bull Call Spread = Long Call (K1, P1) + Short Call (K2, P2) where K2 > K1
    # payoff_long_call_k1 = max(0, stock_price_at_expiration - K1) - P1
    # payoff_short_call_k2 = P2 - max(0, stock_price_at_expiration - K2)
    # return payoff_long_call_k1 + payoff_short_call_k2
    else:
        # print(f"Warning: Payoff for strategy type '{strat_type}' not implemented.")
        return 0

def generate_and_plot_payoff_diagram(strategy_details_list, current_spot_price):
    """
    Generates and plots a P/L diagram for a list of strategy components or a single strategy.
    
    Args:
        strategy_details_list (list of dict): A list where each dict defines one leg of a strategy 
                                             (or a single dict if one leg strategy).
                                             Example: [{'type': 'long_call', 'strike': 100, 'premium': 2.0}]
                                             For a spread: [{'type': 'long_call', 'strike': 100, 'premium': 2.0}, 
                                                            {'type': 'short_call', 'strike': 105, 'premium': 1.0}]
        current_spot_price (float): The current spot price of the underlying, used to center the plot range.
    """
    plt.ioff() # Turn off interactive mode for Matplotlib as a precaution
    if not strategy_details_list:
        print("No strategy details provided for plotting.")
        return

    # Determine a reasonable range for stock prices on the x-axis
    # e.g., 20-30% around the current_spot_price or involved strikes
    min_strike = min(leg['strike'] for leg in strategy_details_list) if strategy_details_list else current_spot_price
    max_strike = max(leg['strike'] for leg in strategy_details_list) if strategy_details_list else current_spot_price
    
    plot_center = current_spot_price
    if min_strike != max_strike: # If it's a spread, center around the strikes or mid-point
        plot_center = (min_strike + max_strike) / 2
    
    price_range_percentage = 0.30 # Plot 30% around the center
    min_x = plot_center * (1 - price_range_percentage)
    max_x = plot_center * (1 + price_range_percentage)
    stock_prices_x = np.linspace(min_x, max_x, 100)
    
    total_payoff_y = np.zeros_like(stock_prices_x)
    
    plot_title = "Payoff Diagram: "
    leg_titles = []

    for leg_details in strategy_details_list:
        leg_payoff_y = np.array([plot_payoff(leg_details, s_price) for s_price in stock_prices_x])
        total_payoff_y += leg_payoff_y
        leg_titles.append(f"{leg_details['type'].replace('_', ' ').title()} K={leg_details['strike']} (Cost/Credit: {leg_details['premium']:.2f})") # Positive for cost, negative for credit if structured that way
        
    if len(strategy_details_list) == 1:
        plot_title += leg_titles[0]
    else:
        # Create a name for the spread if multiple legs
        # This is a very basic naming convention
        if len(strategy_details_list) == 2:
            type1, strike1 = strategy_details_list[0]['type'], strategy_details_list[0]['strike']
            type2, strike2 = strategy_details_list[1]['type'], strategy_details_list[1]['strike']
            if type1 == 'long_call' and type2 == 'short_call' and strike1 < strike2: plot_title += f"Bull Call Spread {strike1}/{strike2}"
            elif type1 == 'long_put' and type2 == 'short_put' and strike1 > strike2: plot_title += f"Bull Put Spread {strike2}/{strike1}"
            elif type1 == 'short_call' and type2 == 'long_call' and strike1 < strike2: plot_title += f"Bear Call Spread {strike1}/{strike2}"
            elif type1 == 'short_put' and type2 == 'long_put' and strike1 > strike2: plot_title += f"Bear Put Spread {strike2}/{strike1}"
            else: plot_title += "Custom Spread"
        else:
            plot_title += "Complex Strategy"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_prices_x, total_payoff_y, label=plot_title, color='blue')
    ax.axhline(0, color='black', lw=0.5) # Zero P/L line
    ax.axvline(current_spot_price, color='grey', linestyle='--', label=f'Current Spot: {current_spot_price:.2f}')
    
    # Mark strike prices
    for leg in strategy_details_list:
        ax.axvline(leg['strike'], color='red', linestyle=':', lw=0.8, label=f"Strike {leg['strike']}" if len(strategy_details_list) > 0 else "Strike")

    ax.set_xlabel("Stock Price at Expiration")
    ax.set_ylabel("Profit / Loss")
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True)
    print("DEBUG: generate_and_plot_payoff_diagram - Returning FIG object. File saving should be disabled.")
    return fig # Return the figure object for Streamlit to display

# --- Main Streamlit App Logic ---
st.set_page_config(layout="wide", page_title="Options Analysis App", initial_sidebar_state="expanded")
st.title("📈 Options Analysis & Strategy App")

# Display Polygon status in sidebar
if POLYGON_ENABLED:
    st.sidebar.success("Polygon.io Client: Initialized Successfully")
elif POLYGON_ERROR_MESSAGE:
    st.sidebar.error(f"Polygon.io Client: {POLYGON_ERROR_MESSAGE}")
else:
    st.sidebar.warning("Polygon.io Client: Not initialized (API key likely missing or invalid in polygon_client.py)")

# Custom CSS to ensure consistent font and revert to default background
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stReportViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    }
    /* Target common text elements for a base font size */
    p, li, [data-testid="stMarkdownContainer"] div, .stAlert div[data-testid="stMarkdownContainer"] {
        font-size: 1rem; /* Adjust as needed, 1rem is often a good default */
        line-height: 1.6;
    }
    /* Ensure headers use the same font family but retain size hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    }
    /* For st.info, st.success etc. (alerts) */
    .stAlert {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    }
    /* Remove previous background overrides to go back to default Streamlit theme */
    [data-testid="stAppViewContainer"],
    [data-testid="stReportViewContainer"] {
        /* background-color: transparent !important; */ /* Let Streamlit default theme control this */
        /* color: inherit !important; */ /* Let Streamlit default theme control this */
    }
    [data-testid="stSidebar"] {
        /* background-color: transparent !important; */ /* Let Streamlit default theme control this */
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("User Input")
# Add a global variable USE_ALPHA_VANTAGE_ACTUALLY and a checkbox to toggle it
# This is a workaround as the global var might not be easily changed by a checkbox if functions are already defined
# A better way is to pass this as a parameter or use session state.
# For now, keeping it simple as a global const. If you want to enable AV, change it at the top of the script.
st.sidebar.info(f"Alpha Vantage Calls: {'ENABLED' if USE_ALPHA_VANTAGE_ACTUALLY else 'DISABLED (API key might be demo/placeholder)'}")
st.sidebar.caption("To enable Alpha Vantage, set USE_ALPHA_VANTAGE_ACTUALLY=True at the script's start if your key is valid.")

stock_ticker_input = st.sidebar.text_input("Enter Stock Ticker:", value="NFLX").upper()

# Add selection for options expiration
# This needs to be populated after fetching options_yf
# We'll handle this dynamically below.

if 'analysis_summary' not in st.session_state:
    st.session_state.analysis_summary = {}
if 'latest_indicators' not in st.session_state:
    st.session_state.latest_indicators = {}
if 'current_stock_price' not in st.session_state:
    st.session_state.current_stock_price = None
if 'options_yf' not in st.session_state:
    st.session_state.options_yf = None
if 'selected_expiry' not in st.session_state:
    st.session_state.selected_expiry = None

if st.sidebar.button("Analyze Ticker", key="analyze_button"):
    st.session_state.analysis_summary = {} 
    st.session_state.latest_indicators = {}
    st.session_state.current_stock_price = None
    st.session_state.options_yf = None
    st.session_state.selected_expiry = None

    if not stock_ticker_input:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner(f"Fetching data for {stock_ticker_input}..."):
            # --- Fetch Current Stock Price (Prioritize Polygon, then Finnhub, then Yahoo Finance) ---
            current_price_source = "N/A"
            st.session_state.current_stock_price = None # Reset before fetching

            if POLYGON_ENABLED and polygon_rest_client:
                poly_price = get_current_stock_price_polygon(stock_ticker_input)
                if poly_price is not None:
                    st.session_state.current_stock_price = poly_price
                    current_price_source = "Polygon.io"
                    print(f"Price for {stock_ticker_input} from Polygon.io: {st.session_state.current_stock_price}")
                else:
                    st.warning(f"Could not fetch current price from Polygon.io for {stock_ticker_input}. Trying Finnhub.")
            
            if st.session_state.current_stock_price is None and FINNHUB_ENABLED and finnhub_client:
                fh_quote = get_stock_quote_finnhub(stock_ticker_input)
                if fh_quote and fh_quote.get('current_price') is not None:
                    st.session_state.current_stock_price = fh_quote['current_price']
                    st.session_state.latest_indicators['finnhub_quote'] = fh_quote 
                    current_price_source = "Finnhub"
                    print(f"Price for {stock_ticker_input} from Finnhub: {st.session_state.current_stock_price}")
                else:
                    st.warning(f"Could not fetch current price from Finnhub for {stock_ticker_input}. Trying Yahoo Finance.")
            
            if st.session_state.current_stock_price is None: # Fallback to Yahoo Finance
                yf_stock_data = get_stock_info_yf(stock_ticker_input)
                if yf_stock_data and yf_stock_data["current_price"]:
                    st.session_state.current_stock_price = yf_stock_data["current_price"]
                    current_price_source = "Yahoo Finance"
                    print(f"Price for {stock_ticker_input} from Yahoo Finance: {st.session_state.current_stock_price}")
                else:
                    st.error(f"Could not determine current price for {stock_ticker_input} from any source.")
                    st.stop()
            
            st.session_state.latest_indicators['current_price'] = st.session_state.current_stock_price
            st.session_state.latest_indicators['price_source'] = current_price_source

            # --- Yahoo Finance Data (Options and Historical) ---
            # Fetch historical prices for TIs (remains yfinance for now)
            historical_prices_yf = get_historical_prices_yf(stock_ticker_input, period="1y")
            if historical_prices_yf is not None and not historical_prices_yf.empty:
                if 'Close' in historical_prices_yf.columns:
                    if len(historical_prices_yf['Close']) >= 20:
                        st.session_state.latest_indicators['sma_20_pandas'] = historical_prices_yf['Close'].rolling(window=20).mean().iloc[-1]
                    if len(historical_prices_yf['Close']) >= 50:
                        st.session_state.latest_indicators['sma_50_pandas'] = historical_prices_yf['Close'].rolling(window=50).mean().iloc[-1]
                    macd_df_pandas = calculate_macd_pandas(historical_prices_yf)
                    if macd_df_pandas is not None and not macd_df_pandas.empty:
                        st.session_state.latest_indicators['macd_pandas'] = macd_df_pandas[['MACD', 'Signal_Line', 'MACD_Histogram']].iloc[-1].to_dict()
            
            # Fetch options data (remains yfinance for now)
            st.session_state.options_yf = get_options_chain_yf(stock_ticker_input)

            # --- Determine Option Expirations (Prioritize Polygon, fallback to yfinance for now) ---
            st.session_state.available_expirations = None
            expirations_source = "N/A"
            if POLYGON_ENABLED:
                print(f"Attempting to fetch expirations from Polygon for {stock_ticker_input}")
                poly_expirations = get_option_expirations_polygon(stock_ticker_input)
                if poly_expirations:
                    st.session_state.available_expirations = poly_expirations
                    expirations_source = "Polygon.io"
                    print(f"Expirations from Polygon: {poly_expirations}")
                else:
                    st.warning(f"Could not fetch expirations from Polygon.io for {stock_ticker_input}. Attempting yfinance.")
            
            if not st.session_state.available_expirations:
                # Fallback or primary if Polygon is disabled/failed
                print(f"Attempting to fetch expirations from yfinance for {stock_ticker_input}")
                # yf_options_data = get_options_chain_yf(stock_ticker_input) # This fetches full chain, we only need expirations here first
                yf_ticker = yf.Ticker(stock_ticker_input)
                raw_yf_expirations = yf_ticker.options
                if raw_yf_expirations:
                    # Use the same cleaning logic as in get_options_chain_yf
                    cleaned_yf_expirations = [clean_date_str(exp_date) for exp_date in raw_yf_expirations]
                    st.session_state.available_expirations = cleaned_yf_expirations
                    expirations_source = "yfinance"
                    print(f"Expirations from yfinance: {cleaned_yf_expirations}")
                    # To keep data flow consistent for now, if yfinance is the source for expirations,
                    # we'll also use it as the source for the option chain data later.
                    # This will be refactored when Polygon chain fetching is complete.
                    st.session_state.options_yf = get_options_chain_yf(stock_ticker_input) # fetch full yf chain if yf is expiry source
                else:
                    st.error(f"Could not fetch option expirations from any source for {stock_ticker_input}.")
                    st.session_state.available_expirations = [] # Ensure it's an empty list to prevent errors

            st.session_state.latest_indicators['expirations_source'] = expirations_source

            # --- Polygon.io Additional Data: Intraday Aggregates & News (if POLYGON_ENABLED) ---
            st.session_state.polygon_hourly_data = None
            st.session_state.polygon_news = None
            st.session_state.latest_indicators['historical_volatility_polygon'] = None # Initialize
            # Initialize Polygon TIs
            st.session_state.latest_indicators['sma_20_polygon'] = None
            st.session_state.latest_indicators['sma_50_polygon'] = None
            st.session_state.latest_indicators['ema_20_polygon'] = None
            st.session_state.latest_indicators['rsi_14_polygon'] = None


            if POLYGON_ENABLED:
                # Fetch last 30 days of hourly data
                print(f"Fetching Polygon hourly aggregates for {stock_ticker_input}")
                today_str = datetime.today().strftime("%Y-%m-%d")
                thirty_days_ago_str = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
                st.session_state.polygon_hourly_data = get_polygon_stock_aggregates(
                    ticker=stock_ticker_input,
                    from_date=thirty_days_ago_str,
                    to_date=today_str,
                    multiplier=1,
                    timespan="hour"
                )
                if st.session_state.polygon_hourly_data:
                    print(f"Fetched {len(st.session_state.polygon_hourly_data)} hourly bars from Polygon.")
                else:
                    print(f"Failed to fetch hourly bars from Polygon for {stock_ticker_input}.")

                # Fetch latest 5 news articles
                print(f"Fetching Polygon news for {stock_ticker_input}")
                st.session_state.polygon_news = get_ticker_news_polygon(stock_ticker_input, limit=5)
                if st.session_state.polygon_news:
                    print(f"Fetched {len(st.session_state.polygon_news)} news articles from Polygon.")
                else:
                    print(f"Failed to fetch news from Polygon for {stock_ticker_input}.")

                # Calculate Historical Volatility using Polygon daily data
                print(f"Calculating Polygon historical volatility for {stock_ticker_input}")
                hv_poly = calculate_historical_volatility_polygon(stock_ticker_input, window=20) # 20-day HV
                if hv_poly is not None:
                    st.session_state.latest_indicators['historical_volatility_polygon'] = hv_poly
                    print(f"Stored Polygon Historical Volatility: {hv_poly:.4f}")
                else:
                    print(f"Failed to calculate historical volatility from Polygon for {stock_ticker_input}.")

                # Calculate SMAs, EMA, RSI using Polygon daily data
                pg_sma20 = calculate_sma_polygon(stock_ticker_input, window=20)
                if pg_sma20 is not None: st.session_state.latest_indicators['sma_20_polygon'] = pg_sma20
                pg_sma50 = calculate_sma_polygon(stock_ticker_input, window=50)
                if pg_sma50 is not None: st.session_state.latest_indicators['sma_50_polygon'] = pg_sma50
                pg_ema20 = calculate_ema_polygon(stock_ticker_input, window=20)
                if pg_ema20 is not None: st.session_state.latest_indicators['ema_20_polygon'] = pg_ema20
                pg_rsi14 = calculate_rsi_polygon(stock_ticker_input, window=14)
                if pg_rsi14 is not None: st.session_state.latest_indicators['rsi_14_polygon'] = pg_rsi14
            
            # --- Alpha Vantage Data (Optional, for other TIs if Polygon not used or fails) ---
            if USE_ALPHA_VANTAGE_ACTUALLY:
                # ... (Alpha Vantage calls remain as they were)
                sma_data_av = get_sma_av(stock_ticker_input, time_period=20)
                if sma_data_av is not None and not sma_data_av.empty: st.session_state.latest_indicators['sma_20_av'] = sma_data_av['SMA'].iloc[0]
                ema_data_av = get_ema_av(stock_ticker_input, time_period=20)
                if ema_data_av is not None and not ema_data_av.empty: st.session_state.latest_indicators['ema_20_av'] = ema_data_av['EMA'].iloc[0]
                rsi_data_av = get_rsi_av(stock_ticker_input, time_period=14)
                if rsi_data_av is not None and not rsi_data_av.empty: st.session_state.latest_indicators['rsi_14_av'] = rsi_data_av['RSI'].iloc[0]
        
        st.success(f"Data fetched for {stock_ticker_input}! Current price source: {current_price_source}")

# --- Display Area ---
if st.session_state.current_stock_price is not None:
    st.header(f"Analysis for: {stock_ticker_input.upper()} (Current Price: ${st.session_state.current_stock_price:.2f} via {st.session_state.latest_indicators.get('price_source', 'N/A')})")

    # Expiration date selection
    if st.session_state.available_expirations: # Use the unified available_expirations
        # expirations = list(st.session_state.options_yf.keys()) # Old way
        expirations = st.session_state.available_expirations
        if not st.session_state.selected_expiry or st.session_state.selected_expiry not in expirations:
             st.session_state.selected_expiry = expirations[0] if expirations else None
        
        current_selection_index = expirations.index(st.session_state.selected_expiry) if st.session_state.selected_expiry in expirations else 0

        st.session_state.selected_expiry = st.selectbox(
            "Select Option Expiration Date:", 
            expirations, 
            index=current_selection_index,
            key=f"expiry_selectbox_{stock_ticker_input}" # Add ticker to key to force re-render on ticker change
            )
    else:
        st.warning("No option expiration dates available to select.")
        st.session_state.selected_expiry = None
    
    if st.session_state.selected_expiry and st.session_state.current_stock_price:
        expiry_to_analyze = st.session_state.selected_expiry
        options_for_selected_expiry = None # This will hold the dict {calls: df, puts: df}
        calls_df_orig, puts_df_orig = None, None # Initialize to None

        # Determine data source for the actual option chain data
        option_chain_source = st.session_state.latest_indicators.get('expirations_source') # Default to where expirations came from

        if POLYGON_ENABLED and option_chain_source == "Polygon.io":
            print(f"Fetching Polygon option chain for {stock_ticker_input}, expiry: {expiry_to_analyze}")
            polygon_chain_data = get_options_chain_polygon(stock_ticker_input, expiry_to_analyze)
            if polygon_chain_data and isinstance(polygon_chain_data.get('calls'), pd.DataFrame) and isinstance(polygon_chain_data.get('puts'), pd.DataFrame):
                options_for_selected_expiry = polygon_chain_data
                st.session_state.options_polygon = polygon_chain_data # Store for potential reuse/debugging
                print(f"Successfully fetched and processed option chain from Polygon.io for {expiry_to_analyze}")
                st.success(f"Option chain data loaded from Polygon.io for {expiry_to_analyze}")
            else:
                st.error(f"Failed to fetch or process valid option chain from Polygon.io for {expiry_to_analyze}. Check logs.")
                # Fallback to yfinance if Polygon chain fetch fails
                if st.session_state.options_yf: 
                    st.warning("Falling back to yfinance for option chain data.")
                    options_for_selected_expiry = st.session_state.options_yf.get(expiry_to_analyze, {})
                    option_chain_source = "yfinance (fallback)"
                else:
                    st.error("yfinance options data also not available for fallback.")
                    option_chain_source = "N/A (Polygon failed, yfinance unavailable)"
        
        # Fallback or primary logic if Polygon not enabled or was not source for expirations
        elif option_chain_source == "yfinance" and st.session_state.options_yf:
            print(f"Using yfinance option chain for {stock_ticker_input}, expiry: {expiry_to_analyze}")
            options_for_selected_expiry = st.session_state.options_yf.get(expiry_to_analyze, {})
            st.info(f"Option chain data loaded from yfinance for {expiry_to_analyze}")
        elif not options_for_selected_expiry: # Catch-all if no data source hit yet
             st.warning(f"No option chain data source determined for {expiry_to_analyze}. Expirations source: {st.session_state.latest_indicators.get('expirations_source')}")
             option_chain_source = "N/A"

        st.session_state.latest_indicators['option_chain_source'] = option_chain_source

        if options_for_selected_expiry:
            calls_df_orig = options_for_selected_expiry.get("calls")
            puts_df_orig = options_for_selected_expiry.get("puts")
            
            # Validate that we have DataFrames before proceeding
            if not isinstance(calls_df_orig, pd.DataFrame) or not isinstance(puts_df_orig, pd.DataFrame):
                st.error(f"Option chain data for {expiry_to_analyze} from {option_chain_source} is not in the expected DataFrame format. Check parsing logic.")
                calls_df_orig, puts_df_orig = pd.DataFrame(), pd.DataFrame() # Empty DFs to prevent errors
            elif calls_df_orig.empty and puts_df_orig.empty:
                st.warning(f"No calls or puts data found for {expiry_to_analyze} from {option_chain_source}. The chain might be empty for this date.")
        else:
            st.warning(f"No options data loaded for {expiry_to_analyze}. Source: {option_chain_source}")
            calls_df_orig, puts_df_orig = pd.DataFrame(), pd.DataFrame() # Ensure empty DFs to prevent downstream errors

        avg_atm_call_iv = pd.NA
        avg_atm_put_iv = pd.NA
        analyzed_calls_df = None
        analyzed_puts_df = None

        current_price_for_analysis = st.session_state.current_stock_price

        if calls_df_orig is not None and not calls_df_orig.empty:
            print(f"Analyzing {len(calls_df_orig)} calls from {option_chain_source} for {expiry_to_analyze}")
            analyzed_calls_df, avg_atm_call_iv = analyze_options_data(
                calls_df_orig, 
                current_price_for_analysis, 
                expiry_to_analyze, 
                'calls',
                option_chain_source_str=option_chain_source # Pass the source
            )
        else:
            print(f"No calls data to analyze for {expiry_to_analyze} from {option_chain_source}")

        if puts_df_orig is not None and not puts_df_orig.empty:
            print(f"Analyzing {len(puts_df_orig)} puts from {option_chain_source} for {expiry_to_analyze}")
            analyzed_puts_df, avg_atm_put_iv = analyze_options_data(
                puts_df_orig, 
                current_price_for_analysis, 
                expiry_to_analyze, 
                'puts',
                option_chain_source_str=option_chain_source # Pass the source
            )
        else:
            print(f"No puts data to analyze for {expiry_to_analyze} from {option_chain_source}")
            
        # Store analyzed data in session state
        st.session_state.analyzed_calls_df = analyzed_calls_df
        st.session_state.analyzed_puts_df = analyzed_puts_df
        st.session_state.latest_indicators['avg_atm_call_iv'] = avg_atm_call_iv
        st.session_state.latest_indicators['avg_atm_put_iv'] = avg_atm_put_iv

    # Technical Analysis Summary
    st.subheader("Technical Analysis Summary")
    tech_summary_md = f"**Current Price ({stock_ticker_input}):** ${st.session_state.latest_indicators.get('current_price', 0):.2f}\n"
    
    # Determine Implied Volatility for display (average of call/put ATM if available)
    avg_atm_call_iv_val = st.session_state.latest_indicators.get('avg_atm_call_iv')
    avg_atm_put_iv_val = st.session_state.latest_indicators.get('avg_atm_put_iv')
    display_avg_atm_iv = pd.NA

    if pd.notna(avg_atm_call_iv_val) and pd.notna(avg_atm_put_iv_val):
        display_avg_atm_iv = (avg_atm_call_iv_val + avg_atm_put_iv_val) / 2
    elif pd.notna(avg_atm_call_iv_val):
        display_avg_atm_iv = avg_atm_call_iv_val
    elif pd.notna(avg_atm_put_iv_val):
        display_avg_atm_iv = avg_atm_put_iv_val
        
    tech_summary_md += f"**Average ATM Implied Vol (Selected Expiry):** {display_avg_atm_iv:.4f}\n" if pd.notna(display_avg_atm_iv) else "**Average ATM Implied Vol (Selected Expiry):** N/A\n"
    
    # Display Historical Volatility from Polygon if available
    hv_poly_val = st.session_state.latest_indicators.get('historical_volatility_polygon')
    tech_summary_md += f"**20-Day Historical Volatility (Polygon):** {hv_poly_val:.4f}\n" if hv_poly_val is not None else "**20-Day Historical Volatility (Polygon):** N/A\n"

    # --- SMA 20 ---
    sma20_poly = st.session_state.latest_indicators.get('sma_20_polygon')
    sma20_av = st.session_state.latest_indicators.get('sma_20_av')
    sma20_pd = st.session_state.latest_indicators.get('sma_20_pandas')
    final_sma20, sma20_source = None, None

    if sma20_poly is not None:
        final_sma20, sma20_source = sma20_poly, "Polygon.io"
    elif sma20_av is not None:
        final_sma20, sma20_source = sma20_av, "Alpha Vantage"
    elif sma20_pd is not None:
        final_sma20, sma20_source = sma20_pd, "yfinance (Pandas)"
    
    if final_sma20 is not None:
        tech_summary_md += f"**20-day SMA ({sma20_source}):** {final_sma20:.2f} - Price is {'ABOVE' if st.session_state.current_stock_price > final_sma20 else 'BELOW' if st.session_state.current_stock_price < final_sma20 else 'AT'}\n"
    else:
        tech_summary_md += "**20-day SMA:** N/A (All sources failed or disabled)\n"
    st.session_state.latest_indicators['final_sma20'] = final_sma20 # Store for strategy use

    # --- SMA 50 ---
    sma50_poly = st.session_state.latest_indicators.get('sma_50_polygon')
    sma50_pd_yf = st.session_state.latest_indicators.get('sma_50_pandas') # yfinance/pandas is only source other than poly
    final_sma50, sma50_source = None, None

    if sma50_poly is not None:
        final_sma50, sma50_source = sma50_poly, "Polygon.io"
    elif sma50_pd_yf is not None:
        final_sma50, sma50_source = sma50_pd_yf, "yfinance (Pandas)"

    if final_sma50:
        tech_summary_md += f"**50-day SMA ({sma50_source}):** {final_sma50:.2f} - Price is {'ABOVE' if st.session_state.current_stock_price > final_sma50 else 'BELOW' if st.session_state.current_stock_price < final_sma50 else 'AT'}\n"
    else:
        tech_summary_md += "**50-day SMA:** N/A (All sources failed or disabled)\n"
    st.session_state.latest_indicators['final_sma50'] = final_sma50 # Store for strategy use
    
    # --- RSI 14 ---
    rsi14_poly = st.session_state.latest_indicators.get('rsi_14_polygon')
    rsi14_av = st.session_state.latest_indicators.get('rsi_14_av')
    final_rsi14, rsi14_source = None, None

    if rsi14_poly is not None:
        final_rsi14, rsi14_source = rsi14_poly, "Polygon.io"
    elif rsi14_av is not None:
        final_rsi14, rsi14_source = rsi14_av, "Alpha Vantage"

    if final_rsi14 is not None:
        rsi_cond = "NEUTRAL"
        if final_rsi14 > 70: rsi_cond = "OVERBOUGHT"
        elif final_rsi14 < 30: rsi_cond = "OVERSOLD"
        tech_summary_md += f"**14-day RSI ({rsi14_source}):** {final_rsi14:.2f} ({rsi_cond})\n"
    else: 
        tech_summary_md += "**14-day RSI:** N/A (Polygon/AV failed or disabled)\n"
    st.session_state.latest_indicators['final_rsi14'] = final_rsi14 # Store for strategy use

    # --- MACD (remains yfinance/Pandas for now, as Polygon doesn't offer direct MACD) ---
    macd_info_val = st.session_state.latest_indicators.get('macd_pandas')
    if macd_info_val:
        macd_cond = "Neutral"
        if macd_info_val['MACD'] > macd_info_val['Signal_Line'] and macd_info_val['MACD_Histogram'] > 0: macd_cond="Bullish"
        elif macd_info_val['MACD'] < macd_info_val['Signal_Line'] and macd_info_val['MACD_Histogram'] < 0: macd_cond="Bearish"
        tech_summary_md += f"**MACD (Pandas):** Line={macd_info_val['MACD']:.2f}, Signal={macd_info_val['Signal_Line']:.2f}, Hist={macd_info_val['MACD_Histogram']:.2f} ({macd_cond})\n"
    st.markdown(tech_summary_md)

    # --- Display Ticker News (from Polygon) ---
    if st.session_state.polygon_news and POLYGON_ENABLED:
        st.markdown("--- ") # Separator
        st.markdown("### Recent News (Polygon.io)")
        for news_item in st.session_state.polygon_news:
            news_title = getattr(news_item, 'title', "N/A")
            news_url = getattr(news_item, 'article_url', "#")
            news_publisher = getattr(news_item, 'publisher', {}).get('name', "N/A")
            news_published_utc = getattr(news_item, 'published_utc', "N/A")
            # Safely format date
            try:
                if news_published_utc != "N/A":
                    news_published_dt = datetime.fromisoformat(news_published_utc.replace('Z', '+00:00'))
                    news_published_display = news_published_dt.strftime("%Y-%m-%d %H:%M %Z")
                else:
                    news_published_display = "N/A"
            except ValueError:
                news_published_display = news_published_utc # show raw if parsing fails

            st.markdown(f"**[{news_title}]({news_url})**")
            st.caption(f"Published by: {news_publisher} on {news_published_display}")
            if hasattr(news_item, 'summary') and getattr(news_item, 'summary'):
                st.markdown(f"> _{getattr(news_item, 'summary')[:200]}..._" if len(getattr(news_item, 'summary')) > 200 else f"> _{getattr(news_item, 'summary')}_" ) 
        st.markdown("--- ")

    # Strategy Recommendations
    st.subheader("Strategy Recommendations")
    # Prepare inputs for recommend_strategies
    # IMPORTANT: For strategy recommendations, use the average IV from the OPTIONS CHAIN of the selected expiry.
    # This is 'display_avg_atm_iv' calculated above for the summary, which correctly reflects option IV.
    
    # Make sure the correct IV is passed to recommend_strategies
    # The `display_avg_atm_iv` calculated earlier for the summary is the one we want for strategies as it's based on the selected options.
    options_implied_volatility_for_strategies = display_avg_atm_iv # This was calculated based on selected expiry's options
    
    strat_input_indicators = {
        'current_price': st.session_state.latest_indicators.get('current_price'),
        'final_sma20': st.session_state.latest_indicators.get('final_sma20'), # Already determined with priority
        'final_sma50': st.session_state.latest_indicators.get('final_sma50'), # Already determined with priority
        'rsi_14_av': st.session_state.latest_indicators.get('final_rsi14'), # Use the final chosen RSI
        'macd_pandas': st.session_state.latest_indicators.get('macd_pandas'), # Remains from yfinance/pandas
        'avg_atm_iv': options_implied_volatility_for_strategies 
    }
    # strat_input_indicators['avg_atm_iv'] = options_implied_volatility_for_strategies # Add the correct IV here
    # strat_input_indicators['final_sma20'] = final_selected_sma_20 # These are now directly assigned above
    # strat_input_indicators['final_sma50'] = final_selected_sma_50
    
    if strat_input_indicators.get('current_price') and selected_expiry_for_strat:
        # Pass the actual ticker symbol string to the function
        strategy_suggestions, specific_plays = recommend_strategies(
            strat_input_indicators, 
            st.session_state.get('analyzed_calls_df'), # Use directly from session state
            st.session_state.get('analyzed_puts_df'),  # Use directly from session state
            st.session_state.current_stock_price, 
            selected_expiry_for_strat,
            stock_ticker_input.upper() # Pass the actual ticker symbol string
        )
        for i, suggestion in enumerate(strategy_suggestions):
            st.markdown(suggestion, unsafe_allow_html=True) # Use markdown for potential bolding
        
        # Display specific plays in a table if available
        if specific_plays:
            st.markdown("### Tradable Play Ideas")
            specific_plays_df = pd.DataFrame(specific_plays)
            # Define desired column order (optional, but good for consistency)
            column_order = ["Strategy", "Ticker", "Expiry", "Leg 1", "Leg 2", "Premium", "Metric", "Rationale"]
            # Reorder DataFrame columns if they exist, otherwise use existing order
            display_columns = [col for col in column_order if col in specific_plays_df.columns]
            st.dataframe(specific_plays_df[display_columns])
        else:
            st.markdown("No specific plays generated based on current criteria.")

    else: st.warning("Not enough data for strategy recommendations (e.g., current price or expiry not set).")

    # Payoff Diagrams
    st.subheader("Payoff Diagrams")
    # Use analyzed data directly from session_state for plotting as well
    analyzed_calls_for_plot = st.session_state.get('analyzed_calls_df') 
    if analyzed_calls_for_plot is not None and not analyzed_calls_for_plot.empty:
        st.markdown("#### Example 1: Long Call")
        example_call_leg = analyzed_calls_for_plot.iloc[0]
        long_call_strat = [{'type': 'long_call', 'strike': example_call_leg['strike'], 'premium': example_call_leg['lastPrice']}]
        fig_lc = generate_and_plot_payoff_diagram(long_call_strat, st.session_state.current_stock_price)
        if fig_lc:
            print(f"DEBUG: Main app - fig_lc object is: {type(fig_lc)}")
            st.pyplot(fig_lc)
            plt.close(fig_lc) # Explicitly close the specific figure after Streamlit uses it

        st.markdown("#### Example 2: Bull Call Spread")
        if len(analyzed_calls_for_plot) >= 2:
            long_leg = analyzed_calls_for_plot.iloc[0]
            short_leg_options = analyzed_calls_for_plot[analyzed_calls_for_plot['strike'] > long_leg['strike']]
            if not short_leg_options.empty:
                short_leg = short_leg_options.iloc[0]
                bcs_strat = [ {'type': 'long_call', 'strike': long_leg['strike'], 'premium': long_leg['lastPrice']},
                              {'type': 'short_call', 'strike': short_leg['strike'], 'premium': short_leg['lastPrice']}]
                fig_bcs = generate_and_plot_payoff_diagram(bcs_strat, st.session_state.current_stock_price)
                if fig_bcs:
                    print(f"DEBUG: Main app - fig_bcs object is: {type(fig_bcs)}")
                    st.pyplot(fig_bcs)
                    plt.close(fig_bcs) # Explicitly close the specific figure after Streamlit uses it
            else: st.write("Not enough distinct call strikes for Bull Call Spread example.")
        
        # Cleanup Matplotlib state after plotting all diagrams in this section
        plt.clf()
        plt.close('all')

    else:
        st.write("No analyzed call options available to demonstrate payoff diagrams.")

else:
    st.info("Enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin.") 