import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
import pandas as pd
from yahoo_fin import stock_info as si
import numpy as np

# get the list of NASDAQ tickers
tickers = si.tickers_nasdaq()

# create an empty dataframe to store the results
results_df = pd.DataFrame(columns=['Ticker', 
                                   'Growth Rate', 
                                   'EPS', 
                                   'P/E Ratio', 
                                   'Closing Price'])

# loop through each ticker and retrieve the relevant data
for ticker in tickers:
    try:
        # retrieve the historical stock prices
        prices_df = si.get_data(ticker, 
                                start_date = '2022-03-22', 
                                end_date = '2023-03-22')

        # calculate the growth rate for the last year
        growth_rate = ((prices_df['close'].iloc[-1] - 
                        prices_df['close'].iloc[0]) / 
                        prices_df['close'].iloc[0])

        qt = si.get_quote_table(ticker)
        # retrieve the current EPS
        eps = qt['EPS (TTM)']

        # retrieve the current P/E ratio
        pe_ratio = qt['PE Ratio (TTM)']

        # retrieve the most recent closing price
        closing_price = prices_df['close'].iloc[-1]

        # add the results to the dataframe
        results_df = pd.concat([results_df, 
                                pd.DataFrame({'Ticker': ticker, 
                                              'Growth Rate': growth_rate, 
                                              'EPS': eps, 
                                              'P/E Ratio': pe_ratio, 
                                              'Closing Price': closing_price}, 
                                             index=[0])], 
                                ignore_index = True)
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")

# write the results to CSV
results_df.to_csv('nasdaq_stock_data.csv', index = False)
