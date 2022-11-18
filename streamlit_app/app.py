# GET LIBRARIES
import streamlit as st
import math
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import genextreme
from random import choices
import plotly.express as px

def get_ticker(STOCK):
    stock = yf.Ticker(STOCK)
    return(stock)

def get_hist_data(stock, start = '2000-01-01'):
    hist_price = stock.history(start = start)
    hist_price['Date'] = hist_price.index
    hist_price['perc_change'] = ((hist_price['Close'] - hist_price['Open']) / 
                                    hist_price['Open'])
                                    
    return(hist_price)

def get_simulation(stock, 
                   option_date,
                   num_samples = 10000,
                   sample_size = 20000, 
                   upper_scale = 0.60, 
                   upper_shape = -0.09, 
                   lower_scale = 0.65, 
                   lower_shape = -0.1,
                   today = datetime.utcnow().date()):
    
    opt_chain = stock.option_chain(option_date)

    date = {"Date": pd.date_range(datetime.today().date(), option_date)}

    dates = pd.DataFrame(data = date)
    dates['wday'] = dates['Date'].dt.dayofweek
    dates = dates.loc[dates['wday'] != 5]
    dates = dates.loc[dates['wday'] != 6]

    num_trading_days = dates.shape[0]
    
    start = today - timedelta(days = num_trading_days)

    start = start.isoformat()
    start_date = datetime.today().date() - timedelta(days = 1)
    start_date = start_date.isoformat()
    
    hist_price = get_hist_data(stock)
    
    perc_pos = (
        sum(hist_price.loc[hist_price['Date'] > start, 'perc_change'] > 0) / 
        len(hist_price.loc[hist_price['Date'] > start, 'perc_change'])
    )

    perc_neg = 1 - perc_pos

    avg = np.mean(hist_price['perc_change'])
    std = np.std(hist_price['perc_change'])

    samp_upper = genextreme.rvs(upper_shape, 
                                loc = avg, 
                                scale = upper_scale * std,
                                size = round(sample_size * perc_pos))

    samp_lower = -1 * genextreme.rvs(lower_shape, 
                                     loc = avg, 
                                     scale = lower_scale * std,
                                     size = round(sample_size * perc_neg))

    samp = np.append(samp_upper, samp_lower)

    sample_col = pd.Series(range(1,num_trading_days+1)).repeat(num_samples)
    dates_col = dates['Date'].repeat(num_samples)
    sample_values = np.array(choices(samp + 1, 
                                     k = num_samples * num_trading_days
                                     )).reshape(num_trading_days, 
                                                num_samples)

    start_price = hist_price.loc[hist_price['Date'] == start_date, 'Close']

    sample_values[0] = start_price

    price_paths = sample_values.cumprod(axis = 0)

    final_prices = price_paths[num_trading_days - 1]
    
    puts = opt_chain.puts
    puts['expiration_date'] = option_date
    puts['strike_minus_last'] = puts['strike'] - ((puts['ask'] + puts['bid']) / 2)

    for strp in puts.strike_minus_last:
        puts.loc[puts['strike_minus_last'] == strp, 'likelihood_below'] = (
            sum(final_prices < strp) / num_samples)
    
    calls = opt_chain.calls
    calls['expiration_date'] = option_date
    calls['strike_plus_last'] = calls['strike'] + ((calls['ask'] + calls['bid']) / 2)

    for strp in calls.strike_plus_last:
        calls.loc[calls['strike_plus_last'] == strp, 'likelihood_above'] = (
            sum(final_prices > strp) / num_samples)
    
    puts = puts[['expiration_date', 'strike', 'strike_minus_last', 
                 'likelihood_below', 'lastPrice',
                 'bid', 'ask', 'change', 'percentChange', 'volume', 
                 'openInterest', 'impliedVolatility', 'inTheMoney', 
                 'contractSize', 'currency', 'contractSymbol', 'lastTradeDate']]
    
    puts = puts.rename(({'contractSymbol': 'Contract Symbol',
                         'lastTradeDate': 'Last Trade Date',
                         'lastPrice': 'Last Price',
                         'percentChange': 'Percent Change',
                         'openInterest': 'Open Interest',
                         'impliedVolatility': 'Implied Volatility',
                         'inTheMoney': 'In the Money?',
                         'contractSize': 'Contract Size',
                         'expiration_date': 'Expiration Date',
                         'likelihood_below': 'Llhd Pr Blw St-LP'}),
                        axis = 'columns')
    
    calls = calls[['expiration_date', 'strike', 'strike_plus_last', 
                   'likelihood_above', 'lastPrice',
                   'bid', 'ask', 'change', 'percentChange', 'volume', 
                   'openInterest', 'impliedVolatility', 'inTheMoney', 
                   'contractSize', 'currency', 'contractSymbol', 'lastTradeDate']]
                        
    calls = calls.rename(({'contractSymbol': 'Contract Symbol',
                         'lastTradeDate': 'Last Trade Date',
                         'lastPrice': 'Last Price',
                         'percentChange': 'Percent Change',
                         'openInterest': 'Open Interest',
                         'impliedVolatility': 'Implied Volatility',
                         'inTheMoney': 'In the Money?',
                         'contractSize': 'Contract Size',
                         'expiration_date': 'Expiration Date',
                         'likelihood_above': 'Llhd Pr Abv St+LP'}),
                        axis = 'columns')
    
    return(puts, calls, final_prices, hist_price)

def price_chart(hist_price, STOCK):
    cutoff = datetime.today().date() - timedelta(days = 365)
    cutoff = cutoff.isoformat()
    
    hist_price = hist_price.loc[hist_price['Date'] > cutoff]
    
    fig = px.line(hist_price, 
                  x = 'Date', 
                  y = 'Close',
                  title = f'{STOCK} Prices for Last Year')
    fig.update_layout(yaxis_tickprefix = '$')
    return(fig)

def box_final_prices(final_prices, current_price):
    fig = px.box(final_prices,
                 x = "final_price",
                 labels = {"final_price": "Final Simulated Price"},
                 title = "Boxplot of Simulated Prices (Current stock price in red)")
    fig.add_vline(x = current_price, line_color = 'firebrick')
    fig.update_layout(xaxis_tickprefix = '$')
    return(fig)

def hist_final_prices(final_prices, current_price):
    fig = px.histogram(final_prices, 
                       x = "final_price", 
                       histnorm = 'probability density',
                       labels = {"final_price": "Final Simulated Price"},
                       title = "Histogram of Simulated Prices (Current stock price in red)")
    fig.add_vline(x = current_price, line_color = 'firebrick')
    fig.update_layout(xaxis_tickprefix = '$')
    return(fig)

def main() -> None:
    
    st.sidebar.subheader('Evaluate What Stock Options?')
    
    STOCK = st.sidebar.text_input('Stock:', 'CMCSA')
    
    st.sidebar.subheader('What Option Expiration Date?')
    
    ticker = get_ticker(STOCK)
    ex_date = list(ticker.options)
    options_selection = st.sidebar.selectbox(
        "Select Expiration Date to View", 
        options = ex_date
    )
    
    current_price = get_hist_data(ticker, datetime.utcnow().date().isoformat())
    price = round(current_price['Close'].item(), 2)
    
    st.header(f'Options Evaluations for {STOCK}')
    st.text(f'Option Expiration Date of {options_selection}')
    st.text(f'Current stock price: ${price}')
    
    puts, calls, final_prices, hist_price = get_simulation(ticker, options_selection)
    
    put_tab, call_tab, price_tab = st.tabs([f'{STOCK} Puts', 
                                            f'{STOCK} Calls', 
                                            f'Simulated {STOCK} Prices'])
    
    with put_tab:
        st.subheader("Put Options Data")
        
        st.dataframe(puts)
    
    with call_tab:
        st.subheader("Call Options Data")
        
        st.dataframe(calls)
        
    with price_tab:
        price_line = price_chart(hist_price, STOCK)
        st.plotly_chart(price_line, use_container_width = True)
        
        final_prices = pd.DataFrame({"final_price": final_prices})
        
        box = box_final_prices(final_prices, price)
        st.plotly_chart(box, use_container_width = True)
        
        hist = hist_final_prices(final_prices, price)
        st.plotly_chart(hist, use_container_width = True)
    
if __name__ == "__main__":
    st.set_page_config(
        "Options Information by Jake Rozran",
        "📊",
        initial_sidebar_state = "expanded",
        layout = "wide",
    )
    main()
