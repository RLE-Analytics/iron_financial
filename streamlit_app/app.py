# GET LIBRARIES
import streamlit as st
import math
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import genextreme
from random import choices
import plotly.express as px

NUM_SAMPLES = 100000

def get_token():
    token = st.secrets.tradier['prod_token']
    
    return(token)

def get_hist_data(symbol, 
                  token,
                  start = '2000-01-01', 
                  end = datetime.today().date().isoformat(),
                  endpoint = 'https://api.tradier.com',
                  path = '/v1/markets/history'):

    response = requests.get(f'{endpoint}{path}',
        params = {'symbol':f'{symbol}', 
                  'interval': 'daily', 
                  'start': '2000-01-01', 
                  'end': f'{datetime.today().date().isoformat()}'},
        headers = {'Authorization': f'Bearer {token}', 
                   'Accept': 'application/json'})
    json_response = response.json()
    
    hist_price = pd.json_normalize(json_response['history']['day'])
    hist_price = hist_price.rename(({'date': 'Date',
                                     'open': 'Open',
                                     'close': 'Close'}),
                                    axis = 'columns')
    hist_price['perc_change'] = ((hist_price['Close'] - hist_price['Open']) / 
                                    hist_price['Open'])
                                    
    return(hist_price)

def get_option_dates(symbol,
                     token,
                     endpoint = 'https://api.tradier.com',
                     path = '/v1/markets/options/expirations'):
    response = requests.get(f'{endpoint}{path}',
        params = {'symbol': f'{symbol}', 
                  'includeAllRoots': 'true'},
        headers = {'Authorization': f'Bearer {token}', 
                   'Accept': 'application/json'})
    
    json_response = response.json()
    dates = json_response['expirations']['date']
    
    return(dates)

def get_option_chain(symbol, 
                     token, 
                     option_date,
                     endpoint = 'https://api.tradier.com',
                     path = '/v1/markets/options/chains'):
    
    response = requests.get(f'{endpoint}{path}',
        params = {'symbol': f'{symbol}', 
                  'expiration': f'{option_date}', 
                  'greeks': 'true'},
        headers = {'Authorization': f'Bearer {token}', 
                   'Accept': 'application/json'})
    json_response = response.json()
    
    chain = pd.json_normalize(json_response['options']['option'])
    
    return(chain)

def get_current_price(symbol, 
                      token,
                      endpoint = 'https://api.tradier.com',
                      path = '/v1/markets/quotes'):

    response = requests.get(f'{endpoint}{path}',
        params = {'symbols': f'{symbol}', 
                  'greeks': 'false'},
        headers = {'Authorization': f'Bearer {token}', 
                   'Accept': 'application/json'})
    
    json_response = response.json()
    price = json_response['quotes']['quote']['last']
    
    return(price)

def get_simulation(symbol, 
                   token,
                   option_date,
                   num_samples = NUM_SAMPLES,
                   sample_size = 20000, 
                   upper_scale = 0.60, 
                   upper_shape = -0.09, 
                   lower_scale = 0.65, 
                   lower_shape = -0.1,
                   today = datetime.utcnow().date()):

    date = {"Date": pd.date_range(datetime.today().date(), option_date)}

    dates = pd.DataFrame(data = date)
    dates['wday'] = dates['Date'].dt.dayofweek
    dates = dates.loc[dates['wday'] != 5]
    dates = dates.loc[dates['wday'] != 6]

    num_trading_days = dates.shape[0]
    
    start = today - timedelta(days = num_trading_days)

    start = start.isoformat()
    start_date = get_latest_day(symbol, datetime.now() - timedelta(days = 1))
    
    hist_price = get_hist_data(symbol, token)
    
    perc_pos = 0.5
    # (
    #     sum(hist_price.loc[hist_price['Date'] > start, 'perc_change'] > 0) / 
    #     len(hist_price.loc[hist_price['Date'] > start, 'perc_change'])
    # )

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

    sample_col = pd.Series(range(1, num_trading_days + 1)).repeat(num_samples)
    dates_col = dates['Date'].repeat(num_samples)
    sample_values = np.array(choices(samp + 1, 
                                     k = num_samples * num_trading_days
                                     )).reshape(num_trading_days, 
                                                num_samples)

    start_price = get_current_price(symbol, token)

    sample_values[0] = start_price

    price_paths = sample_values.cumprod(axis = 0)

    final_prices = price_paths[num_trading_days - 1]
    
    return(final_prices)

def eval_puts(symbol, token, option_date, final_prices):
    
    opt_chain = get_option_chain(symbol, token, option_date)
    
    puts = opt_chain.loc[opt_chain['option_type'] == 'put']
    puts['symbol'] = symbol
    puts['strike_minus_ask'] = puts['strike'] - puts['ask']
    puts['strike_plus_bid'] = puts['strike'] + puts['bid']

    for strp in puts.strike:
        puts.loc[puts['strike'] == strp, 'likelihood_below'] = (
            sum(final_prices < puts.loc[puts['strike'] == strp, 'strike_minus_ask'].item()) / NUM_SAMPLES)
        
        puts.loc[puts['strike'] == strp, 'likelihood_above'] = (
            sum(final_prices > puts.loc[puts['strike'] == strp, 'strike_plus_bid'].item()) / NUM_SAMPLES)
        
        puts.loc[puts['strike'] == strp, 'likelihood_strike_below'] = (
            sum(final_prices < strp) / NUM_SAMPLES)
        
        puts.loc[puts['strike'] == strp, 'likelihood_strike_above'] = (
            sum(final_prices > strp) / NUM_SAMPLES)
        
        puts.loc[puts['strike'] == strp, 'ev_above_ep'] = (
            np.mean(final_prices[final_prices > puts.loc[puts['strike'] == strp, 'strike_plus_bid'].item()])) 
        
        puts.loc[puts['strike'] == strp, 'ev_below_ep'] = (
            np.mean(final_prices[final_prices < puts.loc[puts['strike'] == strp, 'strike_minus_ask'].item()])) 
    
    return(puts)

def reshape_puts(puts):
    puts = puts.rename(({'strike_minus_ask': 'EP Buy',
                         'strike_plus_bid': 'Effective Price (sell)',
                         'last': 'Last Price',
                         'expiration_date': 'Expiration Date',
                         'likelihood_below': 'Llhd Blw EP',
                         'likelihood_above': 'Llhd Abv EP',
                         'ev_below_ep': 'EV Blw EP',
                         'ev_above_ep': 'EV Abv EP',
                         'likelihood_strike_above': 'Llhd Abv Stk',
                         'likelihood_strike_below': 'Llhd Blw Stk'}),
                        axis = 'columns')
    
    puts.fillna(0, inplace = True)
            
    puts['Cost'] = puts['ask'] * -100
    puts['Gain'] = (puts['strike'] - puts['EV Blw EP'] - puts['ask']) * 100
    puts['Llhd Abv EP'] = 1 - puts['Llhd Blw EP']
    
    puts['EV'] = ((puts['Llhd Abv EP'] * puts['Cost']) + 
                              (puts['Llhd Blw EP'] * puts['Gain']))
    
    # ORDER BY MOST VALUABLE
    # GET ALL 100 STOCKS
    
    puts_buy = puts[['symbol',
                     'strike',
                     'ask',
                     'EP Buy',
                     'Llhd Blw EP',
                     'Llhd Abv EP',
                     'EV Blw EP',
                     'Cost',
                     'Gain',
                     'EV']]
    
    puts_buy = puts_buy.loc[puts_buy['EV'] > 0]
    puts_buy = puts_buy.sort_values(by = 'EV', ascending = False)
    
    return(puts_buy)


def get_latest_day(stock, current_date = datetime.now()):
    if ('-USD' in stock or 
            (current_date.date().weekday() >= 1 and 
                current_date.date().weekday() <= 4)):
        current_date = current_date.date()
    else:
        if current_date.date().weekday() == 5:
            current_date = current_date.date() - timedelta(days = 1)
        elif current_date.date().weekday() == 6: 
            current_date = current_date.date() - timedelta(days = 2)
        else: 
            if current_date.hour + (current_date.minute / 60) < 9.5:
                current_date = current_date.date() - timedelta(days = 3)
            else:
                current_date = current_date.date()
    
    return(current_date)



def main() -> None:
    
    st.sidebar.subheader('Evaluate What Stock Options?')
    
    STOCK = 'AAPL'
    
    st.sidebar.subheader('What Option Expiration Date?')
    
    token = get_token()
    ex_date = get_option_dates(STOCK, token)
    options_selection = st.sidebar.selectbox(
        "Select Expiration Date to View", 
        options = ex_date
    )
    
    price = get_current_price(STOCK, token)
    
    st.header(f'Buying Put Options for 10 Largest Market Cap Stocks')
    st.text(f'Option Expiration Date of {options_selection}')
    st.text(f'Current stock price: ${price}')
    
    final_prices = get_simulation(STOCK, token, options_selection)
    puts = eval_puts(STOCK, token, options_selection, final_prices)
    puts_buy = reshape_puts(puts)
    
    st.dataframe(puts_buy)
            
    
if __name__ == "__main__":
    st.set_page_config(
        "Options Information by Jake Rozran",
        "📊",
        initial_sidebar_state = "expanded",
        layout = "wide",
    )
    main()
