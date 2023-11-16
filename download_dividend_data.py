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

total = 0
two_months = 0
one_month = 0

for key in ticker_dates:
    dates = ticker_dates[key]

    for d in dates:
        startDate = d - dt.timedelta(days=6)
        endDate = d + dt.timedelta(days=1)
        if (startDate.month != endDate.month):
            print('ticker: {}, ex_dividend_date: {}'.format(key, d))
            av.saveDataForDividendData(ticker=key, ex_dividend_date=d)
            exit()

print('one month: {}'.format(one_month))
print('two months: {}'.format(two_months))
print('total: {}'.format(total))

datelist = pd.date_range(dt.date(2022, 10, 10), periods = 10).tolist()

test = dt.date(2022, 10, 30)
av.getData(test, demo=True)