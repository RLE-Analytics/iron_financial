#!/usr/bin/env python3

import pandas as pd
import math
import requests
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import genextreme
from random import choices

def get_token(path_to_config = './'):
    config = pd.read_csv(f'{path_to_config}sandbox.config')
    token = config['prod_token'].iloc[0]

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
    hist_price['Open'] = hist_price['Open'].astype('float64')
    hist_price['Close'] = hist_price['Close'].astype('float64')
    hist_price['perc_change'] = ((hist_price['Close'] - hist_price['Open']) /
                                    hist_price['Open'])

    return(hist_price)


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
                   finish_date,
                   num_samples = 10000,
                   sample_size = 20000,
                   upper_scale = 0.60,
                   upper_shape = -0.09,
                   lower_scale = 0.65,
                   lower_shape = -0.1,
                   today = datetime.today().date()):

    date = {"Date": pd.date_range(today, finish_date)}

    dates = pd.DataFrame(data = date)
    dates['wday'] = dates['Date'].dt.dayofweek
    dates = dates.loc[dates['wday'] != 5]
    dates = dates.loc[dates['wday'] != 6]

    num_trading_days = dates.shape[0]

    start = today - timedelta(days = num_trading_days)
    start = start.isoformat()

    hist_price = get_hist_data(symbol, token)

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
    del samp_upper, samp_lower

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
    del price_paths

    prob_below = sum(final_prices < start_price) / num_samples
    quantiles = np.quantile(final_prices, (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0))

    return(prob_below, quantiles, start_price)

def get_stock_table(url = 'https://iocalc.com/api:gPr_YnZX/getstocks'):
    response = requests.get(url)

    stock_table = pd.json_normalize(response.json()['stocks'])

    return(stock_table)

def post_to_xano(dat,
                 url = 'https://iocalc.com/api:gPr_YnZX/probability_price_change'):

    for record in range(0, dat.shape[0]):
        response = requests.post(url,
                                 params = {'symbol': dat['symbol'].iloc[record],
                                           'sim_length': dat['sim_length'].iloc[record],
                                           'sim_end_date': dat['sim_end_date'].iloc[record],
                                           'current_price': dat['current_price'].iloc[record],
                                           'prob_below_current': dat['prob_below_current'].iloc[record],
                                           'min_sim_price': dat['min_sim_price'].iloc[record],
                                           'percentile_5': dat['percentile_5'].iloc[record],
                                           'percentile_10': dat['percentile_10'].iloc[record],
                                           'percentile_25': dat['percentile_25'].iloc[record],
                                           'median_sim_price': dat['median_sim_price'].iloc[record],
                                           'percentile_75': dat['percentile_75'].iloc[record],
                                           'percentile_90': dat['percentile_90'].iloc[record],
                                           'percentile_95': dat['percentile_95'].iloc[record],
                                           'max_sim_price': dat['max_sim_price'].iloc[record]})

    return(0)

def get_date(num_days,
             today = datetime.today().date(),
             max_days = 90):
    finish_date = today + timedelta(days = max_days + 30)
    date = {"Date": pd.date_range(today, finish_date)}

    dates = pd.DataFrame(data = date)
    dates['wday'] = dates['Date'].dt.dayofweek
    dates = dates.loc[dates['wday'] != 5]
    dates = dates.loc[dates['wday'] != 6]
    dates.reset_index(drop = True, inplace = True)
    
    sim_end_date = dates['Date'].iloc[num_days]
    
    return(sim_end_date.iso_format())

def create_data(days = [5, 10, 15, 20, 30, 60, 90]):

    token = get_token()
    stock_table = get_stock_table()

    prob_change = pd.DataFrame()

    for num_days in days:
        end_date = get_date(num_days)

        for symbol in stock_table['symbol']:

            prob_below_current, quantile, current_price = get_simulation(symbol, token, end_date)

            tmp = pd.DataFrame({'symbol': symbol,
                                'sim_length': num_days,
                                'sim_end_date': end_date,
                                'current_price': current_price,
                                'prob_below_current': prob_below_current,
                                'min_sim_price': quantile[0],
                                'percentile_5': quantile[1],
                                'percentile_10': quantile[2],
                                'percentile_25': quantile[3],
                                'median_sim_price': quantile[4],
                                'percentile_75': quantile[5],
                                'percentile_90': quantile[6],
                                'percentile_95': quantile[7],
                                'max_sim_price': quantile[8]},
                              index = [0])

            prob_change = pd.concat([prob_change, tmp], axis = 0, ignore_index = True)

    return(prob_change)
