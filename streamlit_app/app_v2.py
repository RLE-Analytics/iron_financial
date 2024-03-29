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
                   num_samples = 10000,
                   sample_size = 20000, 
                   upper_scale = 0.60, 
                   upper_shape = -0.09, 
                   lower_scale = 0.65, 
                   lower_shape = -0.1,
                   today = datetime.utcnow().date()):
    
    opt_chain = get_option_chain(symbol, token, option_date)

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
    
    puts = opt_chain.loc[opt_chain['option_type'] == 'put']
    puts['strike_minus_last'] = puts['strike'] - puts['ask']

    for strp in puts.strike:
        puts.loc[puts['strike'] == strp, 'likelihood_below'] = (
            sum(final_prices < puts.loc[puts['strike'] == strp, 'strike_minus_last'].item()) / num_samples)
        
        puts.loc[puts['strike'] == strp, 'likelihood_strike'] = (
            sum(final_prices < strp) / num_samples)
    
    calls = opt_chain.loc[opt_chain['option_type'] == 'call']
    calls['strike_plus_last'] = calls['strike'] + calls['ask']

    for strp in calls.strike:
        calls.loc[calls['strike'] == strp, 'likelihood_above'] = (
            sum(final_prices > calls.loc[calls['strike'] == strp, 'strike_plus_last'].item()) / num_samples)
            
        calls.loc[calls['strike'] == strp, 'likelihood_strike'] = (
            sum(final_prices > strp) / num_samples)

    puts = puts.rename(({'strike_minus_last': 'Effective Price',
                         'last': 'Last Price',
                         'expiration_date': 'Expiration Date',
                         'likelihood_below': 'Llhd Blw EP',
                         'likelihood_strike': 'Llhd Blw Stk'}),
                        axis = 'columns')

    calls = calls.rename(({'strike_plus_last': 'Effective Price',
                           'last': 'Last Price',
                           'expiration_date': 'Expiration Date',
                           'likelihood_above': 'Llhd Abv EP',
                           'likelihood_strike': 'Llhd Abv Stk'}),
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

def strike_to_effective_plot(dat, current_price, puts = True):
    if puts:
        dat['ask'] = dat['ask'] * -1
    
    dat['Strike Price'] = dat['strike']
    
    fig = px.line(dat, x = 'Strike Price', y = 'Effective Price')
    fig = fig.add_bar(x = dat['Strike Price'],
                      y = dat['Strike Price'])
    fig = fig.add_bar(x = dat['Strike Price'],
                      y = dat['ask'])
    fig.add_hline(y = current_price, line_color = 'firebrick')
    fig.update_layout(xaxis_tickprefix = '$',
                      yaxis_tickprefix = '$',
                      showlegend = False)
    return(fig)
    

def effective_to_prob(dat, current_price, puts):
    if puts:
        dat['Llhd Blw EP'] = dat['Llhd Blw EP'] * 100
        fig = px.bar(dat, x = 'strike', y = 'Llhd Blw EP')
    else:
        dat['Llhd Abv EP'] = dat['Llhd Abv EP'] * 100
        fig = px.bar(dat, x = 'strike', y = 'Llhd Abv EP')
    fig.add_vline(x = current_price, line_color = 'firebrick')
    fig.update_layout(xaxis_tickprefix = '$',
                      yaxis_ticksuffix = '%')
    return(fig)

def strike_to_prob(dat, current_price, puts):
    if puts:
        plot_dat = dat[['strike', 'Llhd Blw Stk']]
        plot_dat['Llhd Blw Stk'] = plot_dat['Llhd Blw Stk'] * 100
        fig = px.bar(plot_dat, x = 'strike', y = 'Llhd Blw Stk')
    else:
        plot_dat = dat[['strike', 'Llhd Abv Stk']]
        plot_dat['Llhd Abv Stk'] = plot_dat['Llhd Abv Stk'] * 100
        fig = px.bar(plot_dat, x = 'strike', y = 'Llhd Abv Stk')
    fig.add_vline(x = current_price, line_color = 'firebrick')
    fig.update_layout(xaxis_tickprefix = '$',
                      yaxis_ticksuffix = '%')
    return(fig)


def main() -> None:
    
    st.sidebar.subheader('Evaluate What Stock Options?')
    
    STOCK = st.sidebar.text_input('Stock:', 'AAPL')
    
    st.sidebar.subheader('What Option Expiration Date?')
    
    token = get_token()
    ex_date = get_option_dates(STOCK, token)
    options_selection = st.sidebar.selectbox(
        "Select Expiration Date to View", 
        options = ex_date
    )
    
    price = get_current_price(STOCK, token)
    
    st.header(f'Options Evaluations for {STOCK}')
    st.text(f'Option Expiration Date of {options_selection}')
    st.text(f'Current stock price: ${price}')
    
    puts, calls, final_prices, hist_price = get_simulation(STOCK, 
                                                           token,
                                                           options_selection)
    
    put_tab, call_tab, price_tab, hist_tab = st.tabs([f'{STOCK} Puts', 
                                                      f'{STOCK} Calls', 
                                                      f'Simulated {STOCK} Prices',
                                                      f'Historical {STOCK} Prices'])
    
    with put_tab:
        st.subheader("Put Options Data")
        
        st.dataframe(puts)
        
        bar = strike_to_effective_plot(puts, price, True)
        st.plotly_chart(bar)
        
        prob_bar = effective_to_prob(puts, price, True)
        st.plotly_chart(prob_bar)
        
        strike_bar = strike_to_prob(puts, price, True)
        st.plotly_chart(strike_bar)
    
    with call_tab:
        st.subheader("Call Options Data")
        
        st.dataframe(calls)
        
        bar = strike_to_effective_plot(calls, price, False)
        st.plotly_chart(bar)
        
        prob_bar = effective_to_prob(calls, price, False)
        st.plotly_chart(prob_bar)
        
        strike_bar = strike_to_prob(calls, price, False)
        st.plotly_chart(strike_bar)
        
    with price_tab:
        final_prices = pd.DataFrame({"final_price": final_prices})
        
        box = box_final_prices(final_prices, price)
        st.plotly_chart(box, use_container_width = True)
        
        hist = hist_final_prices(final_prices, price)
        st.plotly_chart(hist, use_container_width = True)
    
    with hist_tab:
        price_line = price_chart(hist_price, STOCK)
        st.plotly_chart(price_line, use_container_width = True)
    
if __name__ == "__main__":
    st.set_page_config(
        "Options Information by Jake Rozran",
        "📊",
        initial_sidebar_state = "expanded",
        layout = "wide",
    )
    main()
