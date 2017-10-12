import os

import numpy as np

from data_reader import process

# PAPER: Specifically, we set up the maximum lag of the LSTM to include 10 successive observations
LSTM_WINDOW_SIZE = 10

PREDICTORS = ['returns',
              'sigma',
              'Trend ADVERT',
              'Trend AIRTVL',
              'Trend AUTO',
              'Trend AUTOBY',
              'Trend AUTOFI',
              'Trend BIZIND',
              'Trend BNKRPT',
              'Trend COMLND',
              'Trend COMPUT',
              'Trend CONSTR',
              'Trend CRCARD',
              'Trend DURBLE',
              'Trend EDUCAT',
              'Trend FINPLN',
              'Trend FURNTR',
              'Trend INSUR',
              'Trend INVEST',
              'Trend JOBS',
              'Trend LUXURY',
              'Trend MOBILE',
              'Trend MTGE',
              'Trend RENTAL',
              'Trend RLEST',
              'Trend SHOP',
              'Trend SMALLBIZ',
              'Trend TRAVEL',
              'Trend UNEMPL']

if 'DEBUG' in os.environ:
    PREDICTORS = PREDICTORS[0:5]
    print('DEBUG we truncate the predictors because we dont consider all the trends.')
    print(PREDICTORS)

INPUT_SIZE = len(PREDICTORS)


def chunker(seq, size):
    return [(seq[pos:pos + size], seq[pos + size:pos + size + 1]) for pos in range(0, len(seq), 1)]


def df_to_keras_format(df):
    keras_x = []
    keras_y = []

    for x, y in chunker(df, LSTM_WINDOW_SIZE):
        print('*' * 80)

        # filter on predictors
        x = x[PREDICTORS]
        y = y[PREDICTORS]

        print(x)
        print(y)

        x_new = x.values
        y_new = y['sigma'].values

        if len(x_new) == LSTM_WINDOW_SIZE and len(y_new) == 1:
            keras_x.append(x_new)
            keras_y.append(y_new)

    keras_x = np.array(keras_x)
    keras_y = np.array(keras_y)
    return keras_x, keras_y


def get_trainable_data():
    tr, te, sigma_mean, sigma_std = process()
    print(tr.head())
    print(te.head())

    x_train, y_train = df_to_keras_format(tr)
    x_test, y_test = df_to_keras_format(te)

    return (x_train, y_train), (x_test, y_test), sigma_mean, sigma_std


if __name__ == '__main__':
    get_trainable_data()
