import os
import codebase.av_utils as av
import datetime as dt
import pandas as pd

# Get dates from 6/1/2022
# It seems like news sentiment does not exist before 4.1.2022
months = pd.date_range(start='6/1/2022', 
                      end='12/1/2023', freq='MS').tolist()

tickers = ['COST']

for key in tickers:
    print(key)

    for m in months:
        month = m.strftime('%Y-%m')
        fileName = '{}_{}_NEWS.csv'.format(key, month) 
        filePath = './news_data/' + fileName

        if not os.path.isfile(filePath):
            av.getNews(ticker=key, startDate=m)

            # remove this to stop every time.
            exit()