import os
from random import randint

import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from data_reader import z_score_inv
from next_batch import LSTM_WINDOW_SIZE, INPUT_SIZE
from next_batch import get_trainable_data

DATA_FILE = 'data.npz'
if not os.path.exists(DATA_FILE):
    (x_train, y_train), (x_test, y_test), sigma_mean, sigma_std = get_trainable_data()
    np.savez_compressed('data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        tr_col_mean=sigma_mean, tr_col_std=sigma_std)
else:
    d = np.load(DATA_FILE)
    x_train = d['x_train']
    y_train = d['y_train']
    x_test = d['x_test']
    y_test = d['y_test']
    sigma_mean = d['tr_col_mean']
    sigma_std = d['tr_col_std']

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)


def print_np_arr(x):
    return np.array_repr(x).replace('\n', '')


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class Monitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        np.set_printoptions(precision=6, suppress=True)
        num_values_to_predict = 10

        print('\n\n')
        print('_' * 80)

        # TODO: make it with pandas and better.
        predictions = self.model.predict(x_test)
        pred_sigmas = [z_score_inv(pred, sigma_mean, sigma_std) for pred in predictions.flatten()]
        true_sigmas = [z_score_inv(true, sigma_mean, sigma_std) for true in y_test.flatten()]
        print('MAPE TEST MODEL = {0}'.format(mean_absolute_percentage_error(np.array(true_sigmas),
                                                                            np.array(pred_sigmas))))
        print('MAPE DUMMY MODEL = {0}'.format(mean_absolute_percentage_error(np.array(true_sigmas),
                                                                             np.roll(np.array(true_sigmas), shift=1))))

        r_train_idx = randint(a=0, b=len(x_train) - num_values_to_predict)
        print('pred train  =',
              print_np_arr(self.model.predict(x_train[r_train_idx:r_train_idx + num_values_to_predict]).flatten()))
        print('truth train =', print_np_arr(y_train[r_train_idx:r_train_idx + num_values_to_predict].flatten()))
        r_test_idx = randint(a=0, b=len(x_test) - num_values_to_predict)
        print('pred  test  =',
              print_np_arr(self.model.predict(x_test[r_test_idx:r_test_idx + num_values_to_predict]).flatten()))
        print('truth test  =', print_np_arr(y_test[r_test_idx:r_test_idx + num_values_to_predict].flatten()))
        print('_' * 80)
        print('\n')


m = Sequential()
m.add(LSTM(256, input_shape=(LSTM_WINDOW_SIZE, INPUT_SIZE)))
m.add(Dense(1, activation='linear'))


def loss(y_pred, y_true):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# PAPER: with mean absolute percent error (MAPE) as the objective loss function
# PAPER: The model is trained by the 'Adam' method
m.compile(optimizer=Adam(lr=0.001), loss='mape')  # mape
m.summary()
monitor = Monitor()

# PAPER: with 32 examples in a batch
# PAPER:  This can be achieved after roughly 600 epochs.
m.fit(x_train, y_train,
      validation_split=0.2,
      shuffle=True,
      batch_size=32,
      epochs=600,
      verbose=1,
      callbacks=[monitor])
