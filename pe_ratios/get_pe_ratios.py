import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from yahoo_fin import stock_info as si
import numpy as np

# Define the list of NASDAQ stock symbols
symbols = si.tickers_nasdaq()

# Define a function to get the P/E ratio, EPS, and yesterday's closing price for a given stock symbol
def get_stock_data(symbol):
    quote_table = si.get_quote_table(symbol)
    pe_ratio = quote_table['PE Ratio (TTM)']
    eps = quote_table['EPS (TTM)']
    prev_close = si.get_data(symbol).tail(1)['close'][0]
    return pe_ratio, eps, prev_close

# Create an empty dataframe to store the stock data for each stock
stock_data = pd.DataFrame(columns=['Symbol', 'P/E Ratio', 'EPS', 'Prev Close'])

# Loop through each stock symbol and get its P/E ratio, EPS, and yesterday's closing price
for symbol in symbols:
    try:
        pe_ratio, eps, prev_close = get_stock_data(symbol)
        
        if np.isnan(pe_ratio):
            continue
        
        stock_data = pd.concat([stock_data, 
                                pd.DataFrame({'Symbol': symbol, 
                                              'P/E Ratio': pe_ratio, 
                                              'EPS': eps, 
                                              'Prev Close': prev_close}, 
                                             index = [0])], 
                                ignore_index=True)
        
        print(f"worked: {symbol}")
    
    except:
        print(f"failed: {symbol}")
        continue
    

# Save the stock data to a CSV file
stock_data.to_csv('nasdaq_stock_data.csv', index=False)
