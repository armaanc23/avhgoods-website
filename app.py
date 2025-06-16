import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd # Added for better data handling with Alpha Vantage

ALPHA_VANTAGE_API_KEY = "W4FTROR58IZ6YVYH" # <-- IMPORTANT: REPLACE WITH YOUR KEY

def get_options_chain_yf(ticker_symbol):
    """
    Fetches the options chain for a given stock ticker using Yahoo Finance.
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    options_data = {}
    if not expirations:
        print(f"No options expirations found for {ticker_symbol} via yfinance")
        return None

    for expiry in expirations:
        try:
            options_chain = ticker.option_chain(expiry)
            options_data[expiry] = {
                "calls": options_chain.calls,
                "puts": options_chain.puts
            }
        except Exception as e:
            print(f"Could not fetch options for {ticker_symbol} on {expiry} from yfinance: {e}")
            continue # Skip to next expiration if one fails
    return options_data

def get_current_stock_price_av(ticker_symbol):
    """
    Fetches the latest closing price for a stock using Alpha Vantage.
    """
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        print("Alpha Vantage API key not set. Please set ALPHA_VANTAGE_API_KEY in app.py.")
        return None
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ts.get_quote_endpoint(symbol=ticker_symbol)
        if not data.empty and '05. price' in data.columns:
            return float(data['05. price'].iloc[0])
        else:
            print(f"Could not retrieve current price for {ticker_symbol} from Alpha Vantage. Data: {data}")
            return None
    except Exception as e:
        print(f"Error fetching current price for {ticker_symbol} from Alpha Vantage: {e}")
        return None

def get_sma_av(ticker_symbol, interval='daily', time_period=20, series_type='close'):
    """
    Fetches the Simple Moving Average (SMA) for a stock using Alpha Vantage.
    """
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        print("Alpha Vantage API key not set. Please set ALPHA_VANTAGE_API_KEY in app.py.")
        return None
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
        print(f"Could not determine current price for {ticker_symbol} from yfinance info.")

    return {"info": info, "current_price": current_price}


if __name__ == "__main__":
    stock_ticker = "MSFT" # Example Ticker

    print(f"--- Data for {stock_ticker} ---")

    # --- Yahoo Finance ---
    print("\n--- Yahoo Finance Data ---")
    yf_stock_data = get_stock_info_yf(stock_ticker)
    if yf_stock_data and yf_stock_data["current_price"]:
        print(f"Yahoo Finance - Current Price for {stock_ticker}: {yf_stock_data['current_price']}")
    else:
        print(f"Could not fetch current price for {stock_ticker} from Yahoo Finance.")

    options_yf = get_options_chain_yf(stock_ticker)
    if options_yf:
        first_expiry = next(iter(options_yf)) # Get the first available expiry
        print(f"\nOptions for {first_expiry} (from Yahoo Finance):")
        if not options_yf[first_expiry]["calls"].empty:
            print("\nCalls (first 5):")
            print(options_yf[first_expiry]["calls"].head())
        else:
            print("No calls found for this expiry.")
        if not options_yf[first_expiry]["puts"].empty:
            print("\nPuts (first 5):")
            print(options_yf[first_expiry]["puts"].head())
        else:
            print("No puts found for this expiry.")
    else:
        print(f"No options data found for {stock_ticker} from Yahoo Finance.")

    # --- Alpha Vantage ---
    print("\n--- Alpha Vantage Data ---")
    if ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        current_price_av = get_current_stock_price_av(stock_ticker)
        if current_price_av:
            print(f"Alpha Vantage - Current Price for {stock_ticker}: {current_price_av}")
        else:
            print(f"Could not fetch current price for {stock_ticker} from Alpha Vantage.")

        sma_data_av = get_sma_av(stock_ticker, time_period=20) # 20-day SMA
        if sma_data_av is not None and not sma_data_av.empty:
            print(f"\nAlpha Vantage - Latest 20-day SMA for {stock_ticker}:")
            print(sma_data_av.head(1)) # Display the most recent SMA value
        else:
            print(f"Could not fetch SMA for {stock_ticker} from Alpha Vantage.")
    else:
        print("Alpha Vantage API key not set. Skipping Alpha Vantage calls.")

    # You can add more function calls and analysis here as we build out features. 