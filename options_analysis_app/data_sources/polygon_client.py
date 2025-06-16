from polygon import RESTClient
import os
import pandas as pd # Make sure pandas is imported
from datetime import datetime # For date parsing if needed
import numpy as np # Added for np.log and np.sqrt

# --- Polygon.io Configuration ---
# It's better to use environment variables for API keys in production.
# For now, we'll use the hardcoded key and make it clear.
POLYGON_API_KEY = "u_ElALBbf0p6lXoqELGvcNVFZD44IGf8" # Your provided Polygon API key

polygon_client = None
polygon_api_key_error = None

if POLYGON_API_KEY and POLYGON_API_KEY != "YOUR_POLYGON_API_KEY_HERE":
    try:
        polygon_client = RESTClient(api_key=POLYGON_API_KEY)
        # You can add a test call here if desired, e.g., fetching reference data
        # print("Polygon.io client initialized successfully.")
    except Exception as e:
        polygon_api_key_error = f"Failed to initialize Polygon.io client: {e}"
        polygon_client = None
        # print(polygon_api_key_error)
else:
    polygon_api_key_error = "Polygon API key is missing or a placeholder. Please set it correctly."
    # print(polygon_api_key_error)

# --- Example function (will be expanded) ---
def get_polygon_stock_aggregates(ticker, from_date, to_date, multiplier=1, timespan="day", adjusted=True, sort="asc", limit=5000):
    """
    Fetches aggregate bars for a stock ticker over a given date range from Polygon.io.

    Args:
        ticker (str): The stock ticker symbol.
        from_date (str): The start of the aggregate time window (YYYY-MM-DD).
        to_date (str): The end of the aggregate time window (YYYY-MM-DD).
        multiplier (int): The size of the timespan multiplier (e.g., 1, 5, 15).
        timespan (str): The size of the time window (e.g., 'minute', 'hour', 'day').
        adjusted (bool): Whether the results are adjusted for splits. Default True.
        sort (str): Sort order ('asc' or 'desc'). Default 'asc'.
        limit (int): Max number of base aggregates queried to create Rresults. Max 50000. Default 5000.

    Returns:
        list: A list of aggregate bar objects from Polygon, or None if an error occurs.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for get_polygon_stock_aggregates: {polygon_api_key_error}")
        return None
    try:
        print(f"Fetching Polygon aggregates for {ticker}: {multiplier} {timespan} from {from_date} to {to_date}")
        # The client's get_aggs method should be used.
        # Note: for very long ranges with small timespans (e.g., 1 minute over years), 
        # the `all_pages=True` and `max_pages` parameters for `list_aggregates` might be needed, 
        # or manual pagination if `get_aggs` itself doesn't support it directly in older client versions.
        # The RESTClient typically has `get_aggs` directly.
        
        # For broad compatibility, let's assume polygon_client refers to the main RESTClient instance.
        aggs_resp = polygon_client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date, # Parameter name in client is from_
            to=to_date,     # Parameter name in client is to
            adjusted=adjusted,
            sort=sort,
            limit=limit
            # The get_aggs might return an iterator or a list directly based on client version/
            # If it returns a response object with a .results attribute, that needs to be handled.
        )

        # The get_aggs method in recent versions of polygon-python-client directly returns a list of Agg objects
        # or raises an exception. It doesn't usually return a response object needing .results unless raw_response=True is used.
        if isinstance(aggs_resp, list):
            return aggs_resp
        # elif hasattr(aggs_resp, 'results'): # For older client versions or specific raw responses
            # return aggs_resp.results
        else:
            # This case should ideally not be hit with modern clients unless an error occurred that wasn't an exception
            print(f"Unexpected response type from Polygon get_aggs for {ticker}: {type(aggs_resp)}")
            return None

    except Exception as e:
        print(f"Error fetching Polygon stock aggregates for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        if "Forbidden" in str(e) or "auth" in str(e).lower() or "status_code=403" in str(e).lower():
            print("Polygon aggregates: API key permission issue or subscription problem.")
            # Consider raising a specific exception or returning an error object if needed upstream.
        return None

def calculate_historical_volatility_polygon(ticker_symbol, window=20, annualization_factor=252):
    """
    Calculates historical realized volatility for a stock using daily closing prices from Polygon.io.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        window (int): The rolling window period for standard deviation calculation (e.g., 20 days).
        annualization_factor (int): The number of trading days in a year (e.g., 252) for annualizing.

    Returns:
        float: The latest annualized historical volatility, or None if an error occurs or data is insufficient.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for calculate_historical_volatility_polygon: {polygon_api_key_error}")
        return None

    # Determine date range for fetching aggregates
    # We need at least 'window' + 1 data points to calculate 'window' returns.
    # Fetch a bit more to be safe, e.g., window + 50 days of data to ensure enough points after holidays/weekends.
    # Polygon's limit for get_aggs is 5000, which is ample for daily data for many years.
    to_date_dt = datetime.today()
    # Estimate start date: go back window days + some buffer (e.g. 1.5x window in calendar days, or a fixed buffer like 60-90 days)
    # A simpler approach: if window is 20, we need 21 prices. If we request ~40-50 trading days, that's ~2-2.5 months.
    from_date_dt = to_date_dt - pd.Timedelta(days=window + 70) # Fetch more data to ensure 'window' periods
    
    from_date_str = from_date_dt.strftime("%Y-%m-%d")
    to_date_str = to_date_dt.strftime("%Y-%m-%d")

    print(f"Fetching daily aggregates for {ticker_symbol} from {from_date_str} to {to_date_str} for HV calculation.")
    daily_aggs = get_polygon_stock_aggregates(
        ticker=ticker_symbol,
        from_date=from_date_str,
        to_date=to_date_str,
        multiplier=1,
        timespan="day",
        adjusted=True,
        sort="asc", # Ensure ascending order for rolling calculations
        limit=window + 50 # Limit to a reasonable number slightly more than needed
    )

    if not daily_aggs or len(daily_aggs) < window + 1:
        print(f"Insufficient historical data for {ticker_symbol} to calculate volatility (need {window + 1}, got {len(daily_aggs) if daily_aggs else 0}).")
        return None

    try:
        # Extract closing prices. Each 'agg' in daily_aggs should have a 'close' attribute.
        prices = [agg.close for agg in daily_aggs if hasattr(agg, 'close')]
        if len(prices) < window + 1:
            print(f"Insufficient valid closing prices for {ticker_symbol} to calculate volatility after extraction (need {window + 1}, got {len(prices)}).")
            return None

        close_prices = pd.Series(prices)
        
        # Calculate daily log returns
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # Calculate rolling standard deviation of log returns
        # Min_periods ensures we only calculate if we have enough data for the window
        rolling_std_dev = log_returns.rolling(window=window, min_periods=window).std()
        
        if rolling_std_dev.empty or pd.isna(rolling_std_dev.iloc[-1]):
            print(f"Could not calculate rolling standard deviation for {ticker_symbol}. Series might be too short or contain NaNs.")
            return None
            
        # Annualize the volatility
        # The last value in the series is the most recent historical volatility
        annualized_hv = rolling_std_dev.iloc[-1] * np.sqrt(annualization_factor)
        
        print(f"Calculated Annualized HV for {ticker_symbol} ({window}-day): {annualized_hv:.4f}")
        return annualized_hv

    except Exception as e:
        print(f"Error calculating historical volatility for {ticker_symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_sma_polygon(ticker_symbol, window=20, annualization_factor=None, # annualization_factor not used for SMA but kept for signature consistency if needed
                            data_points_needed=None, price_column='close'):
    """
    Calculates Simple Moving Average (SMA) for a stock using daily closing prices from Polygon.io.
    Args:
        ticker_symbol (str): The stock ticker symbol.
        window (int): The rolling window period for SMA (e.g., 20 days).
        data_points_needed (int): How many data points to fetch. If None, defaults to window + buffer.
        price_column (str): Column to use for price ('close', 'open', 'high', 'low', 'vwap').
    Returns:
        float: The latest SMA value, or None if an error occurs or data is insufficient.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for calculate_sma_polygon: {polygon_api_key_error}")
        return None

    fetch_limit = data_points_needed if data_points_needed is not None else window + 30 # Fetch a bit more
    to_date_dt = datetime.today()
    from_date_dt = to_date_dt - pd.Timedelta(days=fetch_limit + 60) # Generous buffer for calendar days vs trading days
    
    from_date_str = from_date_dt.strftime("%Y-%m-%d")
    to_date_str = to_date_dt.strftime("%Y-%m-%d")

    print(f"Fetching daily aggregates for {ticker_symbol} (SMA {window}-day, price col: {price_column}) from {from_date_str} to {to_date_str}.")
    daily_aggs = get_polygon_stock_aggregates(
        ticker=ticker_symbol, from_date=from_date_str, to_date=to_date_str,
        multiplier=1, timespan="day", adjusted=True, sort="asc", limit=fetch_limit
    )

    if not daily_aggs or len(daily_aggs) < window:
        print(f"Insufficient historical data for {ticker_symbol} for SMA {window} (need {window}, got {len(daily_aggs) if daily_aggs else 0}).")
        return None

    try:
        prices = [getattr(agg, price_column, None) for agg in daily_aggs if hasattr(agg, price_column)]
        prices = [p for p in prices if p is not None] # Remove None values if attribute existed but was None

        if len(prices) < window:
            print(f"Insufficient valid '{price_column}' prices for {ticker_symbol} for SMA {window} (need {window}, got {len(prices)}).")
            return None

        price_series = pd.Series(prices)
        sma = price_series.rolling(window=window, min_periods=window).mean().iloc[-1]
        
        if pd.isna(sma):
            print(f"SMA calculation resulted in NaN for {ticker_symbol} (window {window}).")
            return None
        
        print(f"Calculated SMA-{window} for {ticker_symbol} (Polygon): {sma:.2f}")
        return sma
    except Exception as e:
        print(f"Error calculating SMA for {ticker_symbol} (Polygon): {e}")
        return None

def calculate_ema_polygon(ticker_symbol, window=20, data_points_needed=None, price_column='close'):
    """
    Calculates Exponential Moving Average (EMA) for a stock using daily closing prices from Polygon.io.
    Args: (similar to SMA)
    Returns:
        float: The latest EMA value, or None.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for calculate_ema_polygon: {polygon_api_key_error}")
        return None

    fetch_limit = data_points_needed if data_points_needed is not None else window + 60 # EMA needs more historical data
    to_date_dt = datetime.today()
    from_date_dt = to_date_dt - pd.Timedelta(days=fetch_limit + 90)
    
    from_date_str = from_date_dt.strftime("%Y-%m-%d")
    to_date_str = to_date_dt.strftime("%Y-%m-%d")

    print(f"Fetching daily aggregates for {ticker_symbol} (EMA {window}-day, price col: {price_column}) from {from_date_str} to {to_date_str}.")
    daily_aggs = get_polygon_stock_aggregates(
        ticker=ticker_symbol, from_date=from_date_str, to_date=to_date_str,
        multiplier=1, timespan="day", adjusted=True, sort="asc", limit=fetch_limit
    )

    if not daily_aggs or len(daily_aggs) < window: # Technically EMA can be calculated with fewer, but result less stable.
        print(f"Insufficient historical data for {ticker_symbol} for EMA {window} (need at least {window}, got {len(daily_aggs) if daily_aggs else 0}).")
        return None

    try:
        prices = [getattr(agg, price_column, None) for agg in daily_aggs if hasattr(agg, price_column)]
        prices = [p for p in prices if p is not None]

        if len(prices) < 1: # Need at least one price point
            print(f"Insufficient valid '{price_column}' prices for {ticker_symbol} for EMA {window} (got 0).")
            return None

        price_series = pd.Series(prices)
        ema = price_series.ewm(span=window, adjust=False, min_periods=window).mean().iloc[-1]

        if pd.isna(ema):
            print(f"EMA calculation resulted in NaN for {ticker_symbol} (window {window}).")
            return None

        print(f"Calculated EMA-{window} for {ticker_symbol} (Polygon): {ema:.2f}")
        return ema
    except Exception as e:
        print(f"Error calculating EMA for {ticker_symbol} (Polygon): {e}")
        return None

def calculate_rsi_polygon(ticker_symbol, window=14, data_points_needed=None, price_column='close'):
    """
    Calculates Relative Strength Index (RSI) for a stock using daily closing prices from Polygon.io.
    Args: (similar to SMA)
    Returns:
        float: The latest RSI value, or None.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for calculate_rsi_polygon: {polygon_api_key_error}")
        return None

    # RSI needs window + 1 prices to get 'window' changes. Fetch more.
    fetch_limit = data_points_needed if data_points_needed is not None else window + 50 
    to_date_dt = datetime.today()
    from_date_dt = to_date_dt - pd.Timedelta(days=fetch_limit + 70)
    
    from_date_str = from_date_dt.strftime("%Y-%m-%d")
    to_date_str = to_date_dt.strftime("%Y-%m-%d")

    print(f"Fetching daily aggregates for {ticker_symbol} (RSI {window}-day, price col: {price_column}) from {from_date_str} to {to_date_str}.")
    daily_aggs = get_polygon_stock_aggregates(
        ticker=ticker_symbol, from_date=from_date_str, to_date=to_date_str,
        multiplier=1, timespan="day", adjusted=True, sort="asc", limit=fetch_limit
    )

    if not daily_aggs or len(daily_aggs) < window + 1:
        print(f"Insufficient historical data for {ticker_symbol} for RSI {window} (need {window + 1}, got {len(daily_aggs) if daily_aggs else 0}).")
        return None

    try:
        prices = [getattr(agg, price_column, None) for agg in daily_aggs if hasattr(agg, price_column)]
        prices = [p for p in prices if p is not None]

        if len(prices) < window + 1:
            print(f"Insufficient valid '{price_column}' prices for {ticker_symbol} for RSI {window} (need {window + 1}, got {len(prices)}).")
            return None

        price_series = pd.Series(prices)
        delta = price_series.diff(1)
        delta = delta.dropna() # Remove first NaN

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        
        # For smoothed RSI (more common, like used in ewm):
        # avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        # avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()


        if avg_loss.iloc[-1] == 0: # Avoid division by zero if no losses in period
            rsi = 100.0
        elif avg_gain.empty or avg_loss.empty or pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]):
             print(f"Could not calculate avg_gain/avg_loss for RSI for {ticker_symbol} (window {window}).")
             return None
        else:
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs))
        
        if pd.isna(rsi):
            print(f"RSI calculation resulted in NaN for {ticker_symbol} (window {window}).")
            return None

        print(f"Calculated RSI-{window} for {ticker_symbol} (Polygon): {rsi:.2f}")
        return rsi
    except Exception as e:
        print(f"Error calculating RSI for {ticker_symbol} (Polygon): {e}")
        return None

if __name__ == '__main__':
    # This is for testing the client directly
    if polygon_client:
        print("Attempting to fetch Polygon.io data for AAPL...")
        # aggs = get_polygon_stock_aggregates("AAPL") # Old test
        today = datetime.today() # Changed from datetime.date.today()
        one_month_ago = today - pd.Timedelta(days=30)
        
        hourly_aggs = get_polygon_stock_aggregates(
            ticker="AAPL", 
            multiplier=1, 
            timespan="hour", 
            from_date=one_month_ago.strftime("%Y-%m-%d"), 
            to_date=today.strftime("%Y-%m-%d")
        )
        if hourly_aggs:
            print(f"Fetched {len(hourly_aggs)} hourly aggregate bars for AAPL.")
            # for bar in hourly_aggs:
            #     print(bar)
        else:
            print("Failed to fetch hourly aggregates for AAPL.")

        # Test historical volatility calculation
        print("\nTesting Historical Volatility for AAPL...")
        hv_aapl = calculate_historical_volatility_polygon("AAPL", window=20)
        if hv_aapl is not None:
            print(f"AAPL 20-day Historical Volatility (Polygon): {hv_aapl:.4f}")
        else:
            print("Failed to calculate HV for AAPL from Polygon.")

        # Test current price
        # print("\nTesting Current Price for AAPL...")
        # price_aapl = get_current_stock_price_polygon("AAPL")
        # if price_aapl:
        #     print(f"Current AAPL price (Polygon): {price_aapl}")
        # else:
        #     print("Failed to get current AAPL price from Polygon.")
            
        # Test SMA, EMA, RSI calculations
        print("\nTesting Technical Indicators for AAPL (Polygon)...")
        sma_20_aapl = calculate_sma_polygon("AAPL", window=20)
        if sma_20_aapl is not None: print(f"AAPL SMA-20 (Polygon): {sma_20_aapl:.2f}")
        else: print("Failed to calculate SMA-20 for AAPL (Polygon).")

        ema_20_aapl = calculate_ema_polygon("AAPL", window=20)
        if ema_20_aapl is not None: print(f"AAPL EMA-20 (Polygon): {ema_20_aapl:.2f}")
        else: print("Failed to calculate EMA-20 for AAPL (Polygon).")

        rsi_14_aapl = calculate_rsi_polygon("AAPL", window=14)
        if rsi_14_aapl is not None: print(f"AAPL RSI-14 (Polygon): {rsi_14_aapl:.2f}")
        else: print("Failed to calculate RSI-14 for AAPL (Polygon).")

        # Test expirations
        # exps = get_option_expirations_polygon("AAPL")
        # if exps:
        #     print(f"Expirations for AAPL: {exps[:5]}")
        #     if len(exps) > 0:
        #         chain = get_options_chain_polygon("AAPL", exps[0])
        #         if chain and chain['calls'] is not None:
        #             print(f"AAPL Calls for {exps[0]}: {len(chain['calls'])} contracts")
        #             print(chain['calls'].head())
        #         else:
        #             print(f"Could not get chain for {exps[0]}")

    else:
        print(f"Polygon client could not be initialized. Error: {polygon_api_key_error}")

def get_option_expirations_polygon(ticker_symbol):
    """
    Fetches available option expiration dates for a given stock ticker using Polygon.io.
    Args:
        ticker_symbol (str): The underlying stock ticker symbol.
    Returns:
        list: A list of expiration date strings (YYYY-MM-DD), or None if an error occurs.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for get_option_expirations_polygon: {polygon_api_key_error}")
        return None
    
    expirations = set() # Use a set to store unique dates
    try:
        # The get_option_contracts method is part of the reference_client in the polygon library
        # We will fetch all active contracts and then extract unique expiration dates.
        # The API paginates, so we need to handle that.
        
        for contract in polygon_client.list_options_contracts(underlying_ticker=ticker_symbol, expired=False, limit=1000, all_pages=True, max_pages=5):
            # The `all_pages=True` and `max_pages` should handle pagination internally if the client version supports it well.
            # Otherwise, manual pagination with `next_url` would be needed.
            if hasattr(contract, 'expiration_date') and contract.expiration_date:
                expirations.add(contract.expiration_date)
        
        if not expirations:
            print(f"No active option contracts found for {ticker_symbol} from Polygon.io or expiration_date attribute missing.")
            # Attempt with a slightly different approach or parameter if the above yields nothing
            # This is a common endpoint structure for Polygon, might need to use the ReferenceClient explicitly for contracts.
            # Let's try the specific reference client call if the direct one doesn't work as expected.
            # This assumes polygon_client is the top-level RESTClient, which has sub-clients.
            if hasattr(polygon_client, 'reference_client') and hasattr(polygon_client.reference_client, 'get_option_contracts'):
                 for contract_ref in polygon_client.reference_client.get_option_contracts(underlying_ticker=ticker_symbol, expired=False, limit=1000, all_pages=True, max_pages=5):
                     if hasattr(contract_ref, 'expiration_date') and contract_ref.expiration_date:
                        expirations.add(contract_ref.expiration_date)
            elif hasattr(polygon_client, 'get_option_contracts'): # some versions might have it directly on RESTClient
                 for contract_ref in polygon_client.get_option_contracts(underlying_ticker=ticker_symbol, expired=False, limit=1000, all_pages=True, max_pages=5):
                     if hasattr(contract_ref, 'expiration_date') and contract_ref.expiration_date:
                        expirations.add(contract_ref.expiration_date)

        if not expirations:
            return None # No dates found
            
        # Sort the dates (optional, but good for UI)
        sorted_expirations = sorted(list(expirations))
        return sorted_expirations

    except Exception as e:
        print(f"Error fetching option expirations for {ticker_symbol} from Polygon.io: {e}")
        # Detailed error can be helpful for debugging API key issues or network problems
        if "Forbidden" in str(e) or "auth" in str(e).lower():
            print("This might be an API key issue (permissions for options data) or subscription tier.")
        return None 

def get_options_chain_polygon(ticker_symbol, expiration_date_str):
    """
    Fetches the full options chain (calls and puts) for a given stock ticker and expiration date from Polygon.io.
    Transforms the data into Pandas DataFrames compatible with the existing app structure.

    Args:
        ticker_symbol (str): The underlying stock ticker symbol.
        expiration_date_str (str): The option expiration date in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary with keys 'calls' and 'puts', each containing a Pandas DataFrame.
              Returns None if an error occurs or no data is found.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for get_options_chain_polygon: {polygon_api_key_error}")
        return None

    all_contracts = []    
    try:
        print(f"Fetching options chain from Polygon for {ticker_symbol} on {expiration_date_str}")
        # The snapshot_options_chain is the direct method. Max limit is 250, pagination may be needed for very liquid stocks.
        # The polygon-python-client handles pagination with `all_pages=True` and `max_pages` for list_ methods.
        # For snapshot methods, we might need to check the documentation or library source if direct pagination params exist
        # or if we have to manually paginate using `next_url`.
        # For now, let's assume a single call is sufficient or the client handles it via a parameter if available.
        # The `list_options_contracts` used for expirations can also be filtered by expiration_date, which might be more robust for full chain.

        # Let's try list_options_contracts first as it has clear pagination support.
        contracts_iterator = polygon_client.list_options_contracts(
            underlying_ticker=ticker_symbol,
            expiration_date=expiration_date_str,
            # contract_type can be 'call' or 'put' - fetch all then filter, or make two calls.
            # Fetching all is simpler for now.
            expired=False, # Ensure we only get active contracts for that day
            limit=1000,  # Max limit for this endpoint
            all_pages=True,
            max_pages=10 # Limit pages to avoid excessive calls during testing/dev
        )
        
        for contract_snapshot in contracts_iterator:
            all_contracts.append(contract_snapshot)
        
        if not all_contracts:
            print(f"No option contracts found for {ticker_symbol} on {expiration_date_str} using list_options_contracts. Trying snapshot...")
            # Fallback to snapshot if list_options_contracts returns nothing for that specific expiry (unlikely if expiry was found)
            # The /v3/snapshot/options/{underlyingAsset} endpoint can be filtered by expiration_date
            # The python client method is likely `snapshot_options_chain` or similar under `polygon_rest_client.options_client` or `polygon_rest_client` directly
            # Checking common client structures:
            snapshot_data = None
            if hasattr(polygon_client, 'snapshot_options_chain'):
                 snapshot_data = polygon_client.snapshot_options_chain(underlying_asset=ticker_symbol, expiration_date=expiration_date_str, limit=250) # Limit 250 per call
            elif hasattr(polygon_client, 'options_client') and hasattr(polygon_client.options_client, 'get_snapshot'): # Older client structure perhaps
                 # This would typically be for a single contract, not a chain. So this path is less likely.
                 pass 
            
            if snapshot_data and hasattr(snapshot_data, 'results'):
                all_contracts = snapshot_data.results
                # Manual pagination would be needed here if snapshot_data.next_url exists
                # For simplicity, we'll assume the first page (up to 250 contracts) is enough for now.
                # if hasattr(snapshot_data, 'next_url') and snapshot_data.next_url:
                #     print(f"WARNING: More options data available via pagination for {ticker_symbol} {expiration_date_str}, but not yet implemented for snapshot endpoint.")

        if not all_contracts:
            print(f"No option contracts found for {ticker_symbol} on {expiration_date_str} from Polygon.io after trying both methods.")
            return None

        calls_data = []
        puts_data = []

        for contract in all_contracts:
            # Ensure we have the necessary nested structures
            details = getattr(contract, 'details', None)
            day_data = getattr(contract, 'day', None)
            last_trade_data = getattr(contract, 'last_trade', None)
            greeks_data = getattr(contract, 'greeks', None)

            if not details: continue # Skip if no basic contract details

            data_point = {
                'contractSymbol': getattr(details, 'ticker', None), # Polygon option ticker
                'strike': getattr(details, 'strike_price', pd.NA),
                'lastPrice': getattr(last_trade_data, 'price', pd.NA) if last_trade_data else pd.NA,
                'bid': getattr(contract, 'last_quote', {}).get('bid', pd.NA) if hasattr(contract, 'last_quote') else pd.NA, # direct access or through last_quote object
                'ask': getattr(contract, 'last_quote', {}).get('ask', pd.NA) if hasattr(contract, 'last_quote') else pd.NA,
                'change': getattr(day_data, 'change', pd.NA) if day_data else pd.NA,
                'percentChange': getattr(day_data, 'change_percent', pd.NA) if day_data else pd.NA,
                'volume': getattr(day_data, 'volume', 0) if day_data else 0, # Default to 0 if not available
                'openInterest': getattr(contract, 'open_interest', 0) if hasattr(contract, 'open_interest') else 0, # Default to 0
                'impliedVolatility': getattr(contract, 'implied_volatility', pd.NA) if hasattr(contract, 'implied_volatility') else pd.NA,
                'delta': getattr(greeks_data, 'delta', pd.NA) if greeks_data else pd.NA,
                'gamma': getattr(greeks_data, 'gamma', pd.NA) if greeks_data else pd.NA,
                'theta': getattr(greeks_data, 'theta', pd.NA) if greeks_data else pd.NA,
                'vega': getattr(greeks_data, 'vega', pd.NA) if greeks_data else pd.NA,
                'contract_type': getattr(details, 'contract_type', None)
            }
            
            # Ensure required fields for analysis are present, even if as NA
            if data_point['strike'] is pd.NA or data_point['lastPrice'] is pd.NA:
                # print(f"Skipping contract due to missing critical data (strike/lastPrice): {data_point['contractSymbol']}")
                pass # Allow processing for analysis, filtering can happen later

            if data_point['contract_type'] == 'call':
                calls_data.append(data_point)
            elif data_point['contract_type'] == 'put':
                puts_data.append(data_point)

        calls_df = pd.DataFrame(calls_data)
        puts_df = pd.DataFrame(puts_data)

        # Standardize column names if needed, or ensure downstream functions handle these names
        # Current app uses: strike, lastPrice, volume, openInterest, impliedVolatility, delta, gamma, theta, vega
        # The mapping is already done in data_point dictionary.

        # Remove the temporary 'contract_type' column if it exists
        if 'contract_type' in calls_df.columns: calls_df = calls_df.drop(columns=['contract_type'])
        if 'contract_type' in puts_df.columns: puts_df = puts_df.drop(columns=['contract_type'])
        
        return {"calls": calls_df, "puts": puts_df}

    except Exception as e:
        print(f"Error fetching/processing Polygon options chain for {ticker_symbol} on {expiration_date_str}: {e}")
        import traceback
        traceback.print_exc()
        if "Forbidden" in str(e) or "auth" in str(e).lower() or "status_code=403" in str(e).lower():
            print("This is likely an API key permission issue or subscription tier problem with Polygon.io options data.")
            st.error("Failed to fetch from Polygon: API key lacks permission or subscription for options data.")
        return None 

def get_ticker_news_polygon(ticker_symbol, limit=10, order="desc", sort="published_utc"):
    """
    Fetches ticker-specific news from Polygon.io.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        limit (int): Number of news articles to return. Max 1000. Default 10.
        order (str): Order of results ('asc' or 'desc'). Default 'desc' (most recent first).
        sort (str): Sort field ('published_utc', 'ticker'). Default 'published_utc'.

    Returns:
        list: A list of news article objects from Polygon, or None if an error occurs.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for get_ticker_news_polygon: {polygon_api_key_error}")
        return None
    
    try:
        print(f"Fetching news for {ticker_symbol} from Polygon.io (limit {limit})")
        # The RESTClient has `get_ticker_news`.
        # It supports pagination via all_pages=True, but for news, we usually want a fixed number of recent articles.
        news_articles = polygon_client.get_ticker_news(
            ticker=ticker_symbol, 
            limit=limit, 
            order=order, 
            sort=sort,
            # published_utc_gte could be used to get news from last N days, e.g. (datetime.date.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        )
        
        # Similar to get_aggs, recent clients should return a list of NewsArticle objects directly.
        if isinstance(news_articles, list):
            return news_articles
        # elif hasattr(news_articles, 'results'): # For older client/raw response style
            # return news_articles.results
        else:
            print(f"Unexpected response type from Polygon get_ticker_news for {ticker_symbol}: {type(news_articles)}")
            return None

    except Exception as e:
        print(f"Error fetching Polygon news for {ticker_symbol}: {e}")
        import traceback
        traceback.print_exc()
        if "Forbidden" in str(e) or "auth" in str(e).lower() or "status_code=403" in str(e).lower():
            print("Polygon news: API key permission issue or subscription problem.")
        return None 

def get_current_stock_price_polygon(ticker_symbol):
    """
    Fetches the last trade price for a stock ticker from Polygon.io.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        float: The last trade price, or None if an error occurs or no price is found.
    """
    if not polygon_client:
        print(f"Polygon client not initialized for get_current_stock_price_polygon: {polygon_api_key_error}")
        return None
    
    try:
        print(f"Fetching last trade price for {ticker_symbol} from Polygon.io")
        # Note: The actual object returned by get_last_trade might vary slightly by client version.
        # Usually, it's an object with attributes directly, or a dict-like object.
        # Based on docs, results object has a 'p' attribute for price.
        last_trade_response = polygon_client.get_last_trade(ticker_symbol)
        
        # The response structure can be a bit nested or direct depending on the client version and specific endpoint model.
        # Example structure from Polygon docs often shows: { "results": { "p": price_value, ... }, ... }
        # However, the python client might unpack this.
        # Let's check for common ways the price might be nested.
        if hasattr(last_trade_response, 'price'): # Direct attribute after client processing
            return float(last_trade_response.price)
        elif hasattr(last_trade_response, 'results') and hasattr(last_trade_response.results, 'p'):
            return float(last_trade_response.results.p)
        elif isinstance(last_trade_response, dict) and 'results' in last_trade_response and 'p' in last_trade_response['results']:
            return float(last_trade_response['results']['p'])
        elif isinstance(last_trade_response, dict) and 'price' in last_trade_response: # Simpler dict structure
             return float(last_trade_response['price'])
        else:
            # If the structure is different, we might need to inspect `last_trade_response` directly
            # print(f"DEBUG Polygon last_trade_response for {ticker_symbol}: {last_trade_response}")
            # Fallback: attempt to get previous close if last trade isn't found or in expected format
            print(f"Could not find price in last trade for {ticker_symbol}. Trying previous close.")
            prev_close_response = polygon_client.get_previous_close_agg(ticker_symbol)
            # prev_close_response is typically a list of Aggregate objects for previous day.
            if prev_close_response and isinstance(prev_close_response, list) and len(prev_close_response) > 0:
                # The list usually contains one aggregate object for the previous day.
                agg = prev_close_response[0]
                if hasattr(agg, 'close'):
                    print(f"Using previous close for {ticker_symbol}: {agg.close}")
                    return float(agg.close)
            
            print(f"Could not determine current/last price for {ticker_symbol} from Polygon (last trade or prev close).")
            return None
            
    except Exception as e:
        print(f"Error fetching last trade price for {ticker_symbol} from Polygon: {e}")
        import traceback
        traceback.print_exc()
        if "Forbidden" in str(e) or "auth" in str(e).lower() or "status_code=403" in str(e).lower():
            print("Polygon last trade price: API key permission issue or subscription problem.")
        return None 