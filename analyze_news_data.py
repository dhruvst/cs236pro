import pandas as pd
from datetime import datetime

months = pd.date_range(start='6/1/2022', 
                      end='12/1/2022', freq='MS').tolist()

key = 'COST'
sum = 0
numDates = 0

# Counting how many are actually relevant.
totalCount = 0
relevantCount = 0
for m in months:
    month = m.strftime('%Y-%m')
    fileName = '{}_{}_NEWS.csv'.format(key, month) 
    filePath = './news_data/' + fileName

    df = pd.read_csv(filePath)

    inputFormat = '%Y-%m-%d %H:%M:%S'
    dateDateTime = datetime.strptime(df['time_published'].iloc[0], inputFormat)

    dateFormat = '%Y%m%d'
    dateStr = dateDateTime.strftime(dateFormat)
    count = 0
    for index, row in df.iterrows():
        
        dateDateTime = datetime.strptime(row['time_published'], inputFormat)

        if (dateStr == dateDateTime.strftime(dateFormat)):
            count = count + 1
        else:
            sum = sum + count
            numDates = numDates + 1
            count = 1
            dateStr = dateDateTime.strftime(dateFormat)
            # print(dateStr)
        
        totalCount = totalCount + 1
        if (row['relevance_score'] > 0.8):
            relevantCount = relevantCount + 1


    
    sum = sum + count
    numDates = numDates + 1

avg = sum/numDates
relevantPct = relevantCount/totalCount*100
print('Average number of {:.2f} news per day'.format(avg))
print('Out of {} news {} ({:.2f}%) were relevant news'.format(totalCount, relevantCount, relevantPct))