import numpy as np

import requests
import pandas as pd
import json
import datetime as dt

from codebase.api_keys import av_key

from alpha_vantage.timeseries import TimeSeries 
import pandas_market_calendars as mcal

# This function is used by LSTM
def download_data(config):
    ts = TimeSeries(key='demo') #you can use the demo API key for this project, but please make sure to eventually get your own API key at https://www.alphavantage.co/support/#api-key. 
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

def saveDataForDividendData(ticker='AAPL', ex_dividend_date=dt.datetime.today(), interval='15min', demo=False):
    url = 'https://www.alphavantage.co/query'
    params = {}

    d = ex_dividend_date

    # Calendar to get valid market open days
    nyse = mcal.get_calendar('NYSE')

    startDate = d - dt.timedelta(days=6)
    valid_days = nyse.valid_days(start_date=startDate, end_date=d)
    print('startDate valid_days: {}'.format(len(valid_days)))
    i = 7
    while(len(valid_days) < 7):
        startDate = d - dt.timedelta(days=i)
        valid_days = nyse.valid_days(start_date=startDate, end_date=d)
        print('while:startDate valid_days: {}'.format(len(valid_days)))
        i = i + 1
    
    # Note: need an extra day since the time is set at 12AM.
    endDate = d + dt.timedelta(days=2)
    valid_days = nyse.valid_days(start_date=d, end_date=endDate)
    print('endDate valid_days: {}'.format(len(valid_days)))
    i = 3
    while(len(valid_days) < 3):
        endDate = d + dt.timedelta(days=i)
        valid_days = nyse.valid_days(start_date=d, end_date=endDate)
        print('while:endDate valid_days: {}'.format(len(valid_days)))
        i = i + 1

    # Uncomment for testing the nyse calendar
    # valid_days = nyse.valid_days(start_date=startDate, end_date=endDate)
    # print(valid_days)
    # return

    # Setup params
    if demo:
        # Setup params
        params['function'] = 'TIME_SERIES_INTRADAY'
        params['symbol'] = 'IBM'
        params['interval'] = '5min'
        params['apikey'] = 'demo'
    else: 
        params['function'] = 'TIME_SERIES_INTRADAY'
        params['symbol'] = ticker
        params['interval'] = interval
        params['month'] = startDate.strftime('%Y-%m')
        params['outputsize'] = 'full'
        params['apikey'] = av_key    

    r = requests.get(url, params=params)

    # Check and make sure you have the time series data
    search_key = 'Time Series'
    result = dict(filter(lambda item: search_key in item[0], r.json().items()))

    if (len(result) < 1):
        print('Key does not exist.')
        print('url:', r.url)
        print(r.json())
        return
    
    key = list(result.keys())[0]
    print("data length:",  len(r.json()[key]))

    _, header = r.json()
    df = pd.DataFrame.from_dict(r.json()[header], orient='index')
    
    # Clean up column names
    df_cols = [i.split(' ')[1] for i in df.columns]
    df.columns = df_cols

    df = df.sort_index()

    # If the data is across two months.
    if (startDate.month != endDate.month):
        params['month'] = endDate.strftime('%Y-%m')

        r = requests.get(url, params=params)

        # Check and make sure you have the time series data
        search_key = 'Time Series'
        result = dict(filter(lambda item: search_key in item[0], r.json().items()))

        if (len(result) < 1):
            print('Key does not exist.')
            print('url:', r.url)
            print(r.json())
            return
        
        key = list(result.keys())[0]
        print("data length:",  len(r.json()[key]))

        _, header = r.json()
        df2 = pd.DataFrame.from_dict(r.json()[header], orient='index')
        
        # Clean up column names
        df2_cols = [i.split(' ')[1] for i in df2.columns]
        df2.columns = df2_cols

        df = pd.concat([df, df2])
        df = df.sort_index()

    # dateFormat = '%Y-%m-%d %H:%M:%S'
    # a = dt.datetime(2023, 11, 14, 19, 00, 00, 00)
    # c0 = df.index.to_series().between(a.strftime(dateFormat), '2023-11-15')
    
    dateFormat = '%Y-%m-%d'
    c0 = df.index.to_series().between(startDate.strftime(dateFormat), 
                                      endDate.strftime(dateFormat))
    df = df.loc[c0]

    print('length of df: {}'.format(len(df)))
    # print(df.head())

    if not demo:
        m = d.strftime('%Y-%m-%d')
        filename = 'div_data/{s}_{m}.csv'.format(s= params['symbol'], m=m) 
        df.to_csv(filename)

def saveData(ticker='AAPL', demo=False):
    url = 'https://www.alphavantage.co/query'
    params = {}

    # Setup params
    if demo:
        # Setup params
        params['function'] = 'TIME_SERIES_INTRADAY'
        params['symbol'] = 'IBM'
        params['interval'] = '5min'
        params['apikey'] = 'demo'
    else: 
        params['function'] = 'TIME_SERIES_INTRADAY'
        params['symbol'] = 'SPY'
        params['interval'] = '1min'
        params['month'] = '2023-09'
        params['outputsize'] = 'full'
        params['apikey'] = av_key    

    r = requests.get(url, params=params)

    # Check and make sure you have the time series data
    search_key = 'Time Series'
    result = dict(filter(lambda item: search_key in item[0], r.json().items()))

    if (len(result) < 1):
        print('Key does not exist.')
        print('url:', r.url)
        print(r.json())
        return
    
    key = list(result.keys())[0]
    print("data length:",  len(r.json()[key]))

    _, header = r.json()
    df = pd.DataFrame.from_dict(r.json()[header], orient='index')
    
    # Clean up column names
    df_cols = [i.split(' ')[1] for i in df.columns]
    df.columns = df_cols

    df = df.sort_index()

    print(df)

    if demo:
        filename = 'data/{s}_{int}_DEMO.csv'.format(s= params['symbol'], int=params['interval']) 
    else:
        filename = 'data/{s}_{m}.csv'.format(s= params['symbol'], m=params['month']) 

    df.to_csv(filename)



def getNews(ticker='AAPL', demo=False):
    url = 'https://www.alphavantage.co/query'
    params = {}

    if demo:
        params['function'] = 'NEWS_SENTIMENT'
        params['tickers'] = 'AAPL'
        params['apikey'] = 'demo'
    else:
        params['function'] = 'NEWS_SENTIMENT'
        params['tickers'] = ticker
        params['time_from'] = '20230901T0000'
        params['time_to'] = '20230930T2359'
        params['limit'] = 1000
        params['apikey'] = av_key    

    r = requests.get(url, params=params)

    # print(r.json()['feed'][0])

    if 'feed' not in r.json():
        print('No feed for {}'.format(ticker))
        return

    # json_formatted = json.dumps(r.json()['feed'][2], indent=2)

    news_sentiments = []
    for news in r.json()['feed']:
        news_sentiment = []
        time_published = datetime.strptime(news['time_published'], '%Y%m%dT%H%M%S')
        news_sentiment.append(time_published)
        ticker_score = [ns for ns in news['ticker_sentiment'] if ns.get('ticker')==params['tickers']]
        if len(ticker_score) > 1:
            raise RuntimeError('More than one sentiment score.')
        
        news_sentiment.append(ticker_score[0]['relevance_score'])
        news_sentiment.append(ticker_score[0]['ticker_sentiment_score'])
        news_sentiments.append(news_sentiment)

    df = pd.DataFrame(news_sentiments, columns=['time_published', 'relevance_score', 'ticker_sentiment_score'])
    print('Found {} news articles for {}'.format(df.shape[0], params['tickers']))

    if demo:
        filename = 'data/{s}_NEWS_DEMO.csv'.format(s= params['tickers']) 
    else:
        m = dt.strptime(params['time_from'], '%Y%m%dT%H%M%S').strftime('%Y-%m')
        filename = 'data/{s}_{m}_NEWS.csv'.format(s= params['tickers'], m=m) 

    df.to_csv(filename)

if __name__ == "__main__":
    # Uncomment to test saveData
    # saveData(demo=False)

    # Uncomment to test getNews
    # getNews('IBM', demo=False)

    # Uncomment to test getData
    getData(demo=True)

