import pandas as pd
from datetime import datetime, timedelta,date
from collections import defaultdict
import pickle

d = pd.read_pickle(open('codebase/data/d_all_EOD.pkd', 'rb'))
for ticker in d.keys():
    d[ticker] = d[ticker].reset_index(drop = True)

# Find stocks that currently pay dividends
divs = []
for ticker in d.keys():
    df = d[ticker].copy(deep=True)
    df = df[df.date > datetime(2022,11,1)]
    if [x for x in df['divCash'].values if x>0]:
        divs.append(ticker)
print(len(divs))
divs

def record_gains(startdate = datetime(2010,1,1), max_gain = True):
    dicts = []
    for ticker in divs:
        gain_ct = 0
        loss_ct = 0
        gain_tot = 0
        loss_tot = 0
        net = 0
        df = d[ticker]
        df = df[df.date > startdate]
        df1 = df[df['divCash']>0]
        for index, row in df1.iterrows():
            df2 = df.loc[index - 1:index + 1]
            if len(df2) != 3:
                print('incorrect dataframe length: {} for ticker {}'.format(len(df2),ticker))
                continue
            if max_gain:
                change = df2.high.values[2] - df2.low.values[0]
            else:
                highs = df2.high.values
                lows = df2.low.values
                change = (highs[2] - highs[0] - lows[2] + lows[0])/2
            div = df2.divCash.values[1]
            gain = change + div
            if gain > 0:
                gain_ct +=1
                gain_tot += gain
            else:
                loss_ct +=1
                loss_tot += gain
            net += gain
        dicts.append({'ticker':ticker
                  , 'gain_ct': gain_ct
                 , 'gain_tot': gain_tot
                 , 'loss_ct': loss_ct
                 , 'loss_tot': loss_tot
                 , 'net': net})
    return pd.DataFrame(dicts).sort_values('net').reset_index(drop=True)
max_gains = record_gains()

STARTDATE = datetime(2022,10,26)

def calc_spy():
    n_shares = 100
    df = d['SPY'][d['SPY'].date > STARTDATE]
    entry = n_shares * df.iloc[0].adjClose
    print('Original Investment: {}'.format(entry))
    exit = n_shares * df.iloc[-1].adjClose
    profit = exit - entry
    print('Profit: {}%'.format(profit/entry*100))
    return entry

orig_invest = calc_spy()

def create_dc_sched(startdate, max_gain):
    dicts = []
    df_divs = pd.DataFrame()
    for ticker in divs:
        gain_tot = 0
        loss_tot = 0
        net = 0
        df = d[ticker]
        df = df[df.date > startdate]
        df1 = df[df['divCash']>0].copy()
        df1['ticker'] = ticker
        df_divs = pd.concat([df_divs, df1], ignore_index=True)
    return df_divs.sort_values('date')
        
df_divs = create_dc_sched(STARTDATE, True)

### For each day, choose the ticker with the highest historical gains
def trades(start_amount, max_gain=True):
    curr_val = start_amount
    avail_amts = [curr_val / 3] * 3
    total = 0
    curr_avail_index = 0
    d_trades = defaultdict(list)
    d_trade_dates = {}
    for date in df_divs.date.unique():
        if curr_avail_index > 2:
            curr_avail_index = 0
        df1 = df_divs[df_divs.date == date]
        tickers = df1.ticker.values
        d_indexes ={}
        for ticker in tickers:
            d_indexes[ticker] = max_gains[max_gains.ticker == ticker].index[0]    
        selected_ticker = sorted([x for x in d_indexes.items()], key=lambda x:x[1], reverse=True)[0][0]
        df2 = d[selected_ticker]
        idx = df2[df2.date == date].index[0]
        df2 = df2.loc[idx - 1:idx + 1]
        if len(df2) == 3:
            d_trades[selected_ticker].append(date)
            d_trade_dates[date] = selected_ticker
            if max_gain:
                entry_price = df2.low.values[0]
                gain = df2.high.values[2] - entry_price + df2.divCash.values[1]
            else:
                highs = df2.high.values
                lows = df2.low.values
                entry_price = (highs[0] + lows[0]) / 2
                gain = (highs[2] + lows[2]) / 2 - entry_price + df2.divCash.values[1]
            n_shares = avail_amts[curr_avail_index] // entry_price
            gain *= n_shares
            avail_amts[curr_avail_index] += gain
            total += gain
            #print('Number of shares: {}, Gain {}'.format(n_shares,gain))
            #print(avail_amts, sum(avail_amts))
            curr_avail_index += 1
    print('Total gains: {}'.format(total))
    print('Percent return: {}%'.format(total/start_amount * 100))
    if max_gain:
        pickle.dump(d_trades,open('codebase/data/d_trades.pkd','wb'))
        pickle.dump(d_trade_dates,open('codebase/data/d_trade_dates.pkd','wb'))
    return d_trades

d_trades = trades(orig_invest)

N_PRIOR_DAYS = 251

### Create Dataset
def adjust_splits(df):
    df = df.reset_index(drop = True)
    splits = [(datetime.strptime(str(x[0])[:10],'%Y-%m-%d'),x[1]) for x in df[df.splitFactor != 1][['date','splitFactor']].to_records(index=False)]
    df = df[['date','close','high','low']].set_index('date')
    for date, factor in splits:
        df.loc[df.index < date] = df.loc[df.index < date] / factor
    return df

d_dfs = defaultdict(dict)
for ticker, dates in d_trades.items():
    df = adjust_splits(d[ticker])
    dates = [datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') for x in dates]
        
    for date in dates:
        df1 = df.loc[df.index <= date][-N_PRIOR_DAYS:]
        if len(df1) >= N_PRIOR_DAYS:
            d_dfs[ticker][date] = df1
            
pickle.dump(d_dfs,open('codebase/data/d_dfs.pkd','wb'))