from glob import glob

import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.nan)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def read_sp_500():
    spy = pd.read_csv('trends/SP500.csv', parse_dates=True, index_col=0)
    spy['Adj Close Log'] = spy['Adj Close'].apply(np.log)
    spy['Adj Close Log Shift 1'] = spy['Adj Close Log'].shift(1)

    spy['returns'] = spy['Adj Close Log'] - spy['Adj Close Log Shift 1']

    print(spy.head())

    spy.dropna(inplace=True)

    spy.drop('Adj Close Log', 1, inplace=True)
    spy.drop('Adj Close Log Shift 1', 1, inplace=True)

    print('')
    print(spy.head())

    spy['u'] = spy['High'].apply(np.log) - spy['Open'].apply(np.log)
    spy['d'] = spy['Low'].apply(np.log) - spy['Open'].apply(np.log)
    spy['c'] = spy['Adj Close'].apply(np.log) - spy['Open'].apply(np.log)

    print('')
    print(spy.head())

    spy['sigma'] = 0.511 * (spy['u'] - spy['d']) ** 2 - 0.019 * (
        spy['c'] * (spy['u'] + spy['d']) - 2 * spy['u'] * spy['d']) - 0.383 * spy['c'] ** 2

    print('')
    print(spy.head())

    spy.drop('u', 1, inplace=True)
    spy.drop('d', 1, inplace=True)
    spy.drop('c', 1, inplace=True)

    print('')
    print(spy.head())

    # import matplotlib.pyplot as plt
    # spy['sigma'].plot()
    # plt.show()
    return spy


def process_trend(trend):
    trend_name = trend.split('/')[1].split('.')[0]
    t = pd.read_csv(trend, parse_dates=True, index_col=0)
    t = t[['Close']]
    t.columns = ['Trend {}'.format(trend_name.upper())]
    print('Trend [{0}] processed.'.format(trend_name))
    # print(t.head())
    return t


def read_trends():
    trends = glob('trends/*.csv')
    trends = filter(lambda x: 'SP500' not in x and 'spy' not in x, trends)
    trends = sorted(trends)  # reproducibility.
    assert len(trends) == 27, 'You should have 27 trends. Check here https://finance.google.com/finance/domestic_trends'
    trends_df_list = []
    for trend in trends:
        trends_df_list.append(process_trend(trend))
    full_trend_df = pd.concat(trends_df_list, axis=1)
    return full_trend_df


def read_all():
    trends = read_trends()
    sp500 = read_sp_500()
    full_data = sp500.join(trends, how='outer')

    print('-' * 80)
    print(full_data.tail(100))
    print('-' * 80)

    full_data.dropna(inplace=True)

    print('-' * 80)
    print(full_data.tail(100))
    print('-' * 80)
    # print(read_sp_500().join(process_trend(trend), how='outer'))
    return full_data


def split_training_test(df):
    # 19-Oct-2004 to 9-Apr-2012 while the test set ranges from 12-Apr-2012 to 24-Jul-2015
    cutoff_date = '9-Apr-2012'
    training_df = df[:cutoff_date]
    testing_df = df[cutoff_date:]
    return training_df, testing_df


def apply_transform(df):
    # we only support those values. They are given by the paper.


    def z_score(x):
        return (x - np.mean(x)) / np.std(x)

    print(df.head())
    df2 = df.apply(lambda x: z_score(x), axis=0)
    df2['returns'] = df['returns']  # preserve them for now
    df2['sigma'] = df['sigma']
    print(df2.head())

    delta_t = 3
    # df3 = pd.rolling_mean(df2, window=delta_t)
    df3 = df2.rolling(window=delta_t, center=False).mean()

    # for returns (SUM)
    # for volatility (SQUARE - SUM - SQRT)
    df3['returns'] = df2['returns'].rolling(delta_t).apply(np.sum)
    df3['sigma'] = df2['sigma'].apply(np.square).rolling(delta_t).apply(np.sum).apply(np.sqrt)

    df3.dropna(inplace=True)
    return df3


def process():
    df = read_all()
    df = apply_transform(df)
    tr, te = split_training_test(df)
    return tr, te


if __name__ == '__main__':
    r = read_all()
    apply_transform(r)
