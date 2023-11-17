import os
from codebase import av_utils as av

import pandas as pd
import datetime as dt

tickers = pd.read_pickle('codebase/data/dividend_tickers.pkd')
all_ex_dividend_dates = pd.read_pickle('codebase/data/d_all_EOD.pkd')

ticker_dates = {}
for t in tickers:
    df = all_ex_dividend_dates[t]
    dates = df[df.divCash > 0].date.values[-20:]
    dates = [dt.datetime.strptime(str(x)[:10],'%Y-%m-%d') for x in dates]
    ticker_dates[t] = dates

for key in ticker_dates:
    dates = ticker_dates[key]

    for d in dates:
        month = d.strftime('%Y-%m-%d')
        fileName = '{}_{}.csv'.format(key, month) 
        filePath = './div_data/' + fileName

        if not os.path.isfile(filePath):
            # print("{} Ex Dividend Date: {}".format(key, d))
            av.saveDataForDividendData(ticker=key, ex_dividend_date=d)
            
            # remove this to stop every time.
            exit()