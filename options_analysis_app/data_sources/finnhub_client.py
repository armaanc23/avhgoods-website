import finnhub
import os

# IMPORTANT: Store your API key securely, e.g., in environment variables or Streamlit secrets
# For now, using the provided key directly.
# Replace with your own key if necessary, or use st.secrets for deployment.
FINNHUB_API_KEY = "d0ut05hr01qmg3uirbq0d0ut05hr01qmg3uirbqg" # User provided

finnhub_client = None
if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY": # Basic check
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    except Exception as e:
        print(f"Error initializing Finnhub client: {e}")
        finnhub_client = None
else:
    print("Finnhub API key not found or is a placeholder. Finnhub features will be disabled.")

def get_stock_quote_finnhub(ticker_symbol: str):
    """
    Fetches the latest stock quote (current price, high, low, open, previous close) from Finnhub.
    
    Args:
        ticker_symbol (str): The stock ticker symbol.
        
    Returns:
        dict: A dictionary containing quote data (c, h, l, o, pc, t) or None if an error occurs.
              c: Current price
              h: High price of the day
              l: Low price of the day
              o: Open price of the day
              pc: Previous close price
              t: Timestamp
    """
    if not finnhub_client:
        print("Finnhub client not initialized. Cannot fetch quote.")
        return None
    try:
        quote = finnhub_client.quote(ticker_symbol.upper())
        # Finnhub returns {'c': 0, 'd': None, 'dp': None, 'h': 0, 'l': 0, 'o': 0, 'pc': 0, 't': 0} on error for some invalid tickers
        if quote and quote.get('c', 0) != 0: # Basic check for valid data
             # Ensure all expected keys are present, even if we primarily use 'c'
            return {
                'current_price': quote.get('c'),
                'high': quote.get('h'),
                'low': quote.get('l'),
                'open': quote.get('o'),
                'previous_close': quote.get('pc'),
                'timestamp': quote.get('t')
            }
        else:
            print(f"No valid quote data received from Finnhub for {ticker_symbol}. Response: {quote}")
            return None
    except Exception as e:
        print(f"Error fetching quote for {ticker_symbol} from Finnhub: {e}")
        return None

if __name__ == '__main__':
    # Example usage (for testing this client directly)
    sample_ticker = "AAPL"
    if finnhub_client:
        print(f"Attempting to fetch quote for {sample_ticker} using Finnhub...")
        quote_data = get_stock_quote_finnhub(sample_ticker)
        if quote_data:
            print(f"Current Price (Finnhub) for {sample_ticker}: {quote_data.get('current_price')}")
            print(f"Full quote data: {quote_data}")
        else:
            print(f"Failed to get quote data for {sample_ticker} from Finnhub.")
    else:
        print("Finnhub client not set up. Skipping example usage.")

    sample_ticker_invalid = "INVALIDTICKERXYZ"
    if finnhub_client:
        print(f"\nAttempting to fetch quote for {sample_ticker_invalid} using Finnhub...")
        quote_data_invalid = get_stock_quote_finnhub(sample_ticker_invalid)
        if quote_data_invalid:
            print(f"Current Price (Finnhub) for {sample_ticker_invalid}: {quote_data_invalid.get('current_price')}")
            print(f"Full quote data: {quote_data_invalid}")
        else:
            print(f"Failed to get quote data for {sample_ticker_invalid} from Finnhub (as expected for invalid ticker).") 