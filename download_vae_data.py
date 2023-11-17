import os
import codebase.av_utils as av
import datetime as dt
import pandas as pd

# Get dates from 2010
months = pd.date_range(start='1/1/2010', 
                      end='12/1/2022', freq='MS').tolist()

tickers = ['MA', 'GWW', 'EQIX', 'LRCX' 'BLK']

for key in tickers:
    print(key)

    for m in months:
        month = m.strftime('%Y-%m')
        fileName = '{}_{}.csv'.format(key, month) 
        filePath = './vae_data/' + fileName

        if not os.path.isfile(filePath):
            av.saveVAEData(ticker=key, month=month)

            # remove this to stop every time.
            exit()