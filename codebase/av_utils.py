import os
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
        params['extended_hours'] = 'false'
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

def saveVAEData(ticker='AAPL', month='2023-09', interval="15min", demo=False):
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
        params['symbol'] = ticker
        params['interval'] = interval
        params['extended_hours'] = 'false'
        params['month'] = month
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
    # print("data length:",  len(r.json()[key]))

    _, header = r.json()
    df = pd.DataFrame.from_dict(r.json()[header], orient='index')
    
    # Clean up column names
    df_cols = [i.split(' ')[1] for i in df.columns]
    df.columns = df_cols

    df = df.sort_index()

    # print(df)

    if demo:
        filename = 'vae_data/{s}_{int}_DEMO.csv'.format(s= params['symbol'], int=params['interval']) 
    else:
        filename = 'vae_data/{s}_{m}.csv'.format(s= params['symbol'], m=params['month']) 

    df.to_csv(filename)



def saveNews(ticker='AAPL', startDate=dt.datetime(2022, 1, 1, 00, 00, 00, 00), demo=False):
    url = 'https://www.alphavantage.co/query'
    params = {}

    
    endDate = startDate + dt.timedelta(days=30)

    dateFormat = '%Y%m%dT%H%M'
    startDateStr = startDate.strftime(dateFormat)
    endDateStr = endDate.strftime(dateFormat)

    if demo:
        params['function'] = 'NEWS_SENTIMENT'
        params['tickers'] = 'AAPL'
        params['apikey'] = 'demo'
    else:
        params['function'] = 'NEWS_SENTIMENT'
        params['tickers'] = ticker
        params['time_from'] = startDateStr
        params['time_to'] = endDateStr
        params['limit'] = 1000
        params['apikey'] = av_key    

    r = requests.get(url, params=params)

    # print(r.json()['feed'][0])

    if 'feed' not in r.json():
        print('No feed for {}'.format(ticker))
        print(r.json())
        print(r.url)
        return

    # json_formatted = json.dumps(r.json()['feed'][2], indent=2)

    news_sentiments = []
    for news in r.json()['feed']:
        news_sentiment = []
        time_published = dt.datetime.strptime(news['time_published'], '%Y%m%dT%H%M%S')
        news_sentiment.append(time_published)
        ticker_score = [ns for ns in news['ticker_sentiment'] if ns.get('ticker')==params['tickers']]
        if len(ticker_score) > 1:
            raise RuntimeError('More than one sentiment score.')
        
        news_sentiment.append(ticker_score[0]['relevance_score'])
        news_sentiment.append(ticker_score[0]['ticker_sentiment_score'])
        news_sentiment.append(ticker_score[0]['ticker_sentiment_label'])
        news_sentiments.append(news_sentiment)

    df = pd.DataFrame(news_sentiments, columns=['time_published', 'relevance_score', 'ticker_sentiment_score', 'ticker_sentiment_label'])
    print('Found {} news articles for {}'.format(df.shape[0], params['tickers']))

    df = df.sort_values(by=['time_published'])

    if demo:
        filename = 'news_data/{s}_NEWS_DEMO.csv'.format(s= params['tickers']) 
    else:
        m = startDate.strftime('%Y-%m')
        filename = 'news_data/{s}_{m}_NEWS.csv'.format(s= params['tickers'], m=m) 

    df.to_csv(filename)

def getNewsForDates(ticker, dates):
    url = 'https://www.alphavantage.co/query'
    params = {}
    startDate = dates[0]

    d = startDate.strftime('%Y-%m-%d')
    fileName = '{}_{}.csv'.format(ticker, d) 
    filePath = './news_data/' + fileName

    if os.path.isfile(filePath):
        df = pd.read_csv(filePath)
        df = df.set_index('time_published')
        df = df.drop(columns=['ticker_sentiment_label'])
        return df
    
    
    dateFormat = '%Y%m%dT%H%M'
    startDateStr = startDate.strftime(dateFormat)

    endDate = dates[-1] + dt.timedelta(days=1)
    endDateStr = endDate.strftime(dateFormat)

    params['function'] = 'NEWS_SENTIMENT'
    params['tickers'] = ticker
    params['time_from'] = startDateStr
    params['time_to'] = endDateStr
    params['limit'] = 1000
    params['apikey'] = av_key    

    r = requests.get(url, params=params)

    # print(r.json()['feed'][0])

    if 'feed' not in r.json():
        print('No feed for {}'.format(ticker))
        print(r.json())
        print(r.url)

        news_sentiments = []
        for d in dates:
            news_sentiment = {}
            news_sentiment['time_published'] = d
            news_sentiment['relevance_score'] = 0 
            news_sentiment['ticker_sentiment_score'] = 0
            news_sentiment['ticker_sentiment_label'] = 'None'
            news_sentiments.append(news_sentiment)

        return pd.DataFrame(news_sentiments)

    # json_formatted = json.dumps(r.json()['feed'][2], indent=2)
    inputFormat = '%Y%m%dT%H%M%S'
    news_sentiment = {}
    news_sentiments = []
    time_published = dt.datetime.strptime(r.json()['feed'][0]['time_published'], inputFormat)
    
    count = 0
    for news in r.json()['feed']:
        count = count + 1
        ticker_score = [ns for ns in news['ticker_sentiment'] if ns.get('ticker')==params['tickers']]
        if len(ticker_score) > 1:
            raise RuntimeError('More than one sentiment score.')
        
        new_time_published = dt.datetime.strptime(news['time_published'], inputFormat)

        if(time_published.date() == new_time_published.date()):
            if 'relevance_score' not in news_sentiment or ticker_score[0]['relevance_score'] > news_sentiment['relevance_score']:
                news_sentiment['time_published'] = new_time_published
                news_sentiment['relevance_score'] = ticker_score[0]['relevance_score']
                news_sentiment['ticker_sentiment_score'] = ticker_score[0]['ticker_sentiment_score']
                news_sentiment['ticker_sentiment_label'] = ticker_score[0]['ticker_sentiment_label']
        else:
            if pd.to_datetime(news_sentiment['time_published'].date()) in dates:
                news_sentiments.append(news_sentiment)
            time_published = new_time_published
            news_sentiment = {}
            news_sentiment['time_published'] = time_published
            news_sentiment['relevance_score'] = ticker_score[0]['relevance_score']
            news_sentiment['ticker_sentiment_score'] = ticker_score[0]['ticker_sentiment_score']
            news_sentiment['ticker_sentiment_label'] = ticker_score[0]['ticker_sentiment_label']

    if pd.to_datetime(news_sentiment['time_published'].date()) in dates:
        news_sentiments.append(news_sentiment)

    print('Found {} news articles for {}'.format(count, params['tickers']))

    df = pd.DataFrame(news_sentiments)
    df = df.set_index('time_published')

    update = False
    for d in dates:
        if d not in pd.to_datetime(df.index.date):
            news_sentiment = {}
            news_sentiment['time_published'] = d
            news_sentiment['relevance_score'] = 0 
            news_sentiment['ticker_sentiment_score'] = 0
            news_sentiment['ticker_sentiment_label'] = 'None'
            news_sentiments.append(news_sentiment)
            update = True
    
    if update:
        df = pd.DataFrame(news_sentiments)
        df = df.set_index('time_published')

    df = df.sort_index()

    d = startDate.strftime('%Y-%m-%d')
    filename = 'news_data/{s}_{d}.csv'.format(s= params['tickers'], d=d)
    df.to_csv(filename)

    df = df.drop(columns=['ticker_sentiment_label'])

    return df

if __name__ == "__main__":
    # Uncomment to test saveData
    # saveData(demo=False)

    # Uncomment to test getNews
    # getNews('IBM', demo=False)

    # Uncomment to test getData
    getData(demo=True)

