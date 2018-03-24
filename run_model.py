import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from data_reader import z_score_inv
from next_batch import LSTM_WINDOW_SIZE, INPUT_SIZE
from next_batch import get_trainable_data

plt.ion()

DATA_FILE = 'data.npz'
if not os.path.exists(DATA_FILE):
    (x_train, y_train), (x_test, y_test), mean, std = get_trainable_data()
    np.savez_compressed('data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        mean=mean, std=std)
else:
    d = np.load(DATA_FILE)
    x_train = d['x_train']
    y_train = d['y_train']
    x_test = d['x_test']
    y_test = d['y_test']
    mean = d['mean']
    std = d['std']

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

        print('\n\n')
        print('_' * 80)

        # TODO: make it with pandas and better.
        predictions = self.model.predict(x_test)
        # TODO: should be the mean_sigma of std_sigma only
        pred_sigmas = [z_score_inv(pred, mean, std) for pred in predictions.flatten()]
        true_sigmas = [z_score_inv(true, mean, std) for true in y_test.flatten()]
        dummy_sigmas = [z_score_inv(dummy, mean, std) for dummy in np.roll(y_test.flatten(), shift=1)]

        plt.clf()
        plt.plot(true_sigmas, color='blue')
        plt.plot(pred_sigmas, color='lime')
        plt.pause(0.001)
        plt.show()

        print('MAPE TEST MODEL = {0}'.format(mean_absolute_percentage_error(np.array(true_sigmas),
                                                                            np.array(pred_sigmas))))
        print('MAPE DUMMY MODEL = {0}'.format(mean_absolute_percentage_error(np.array(true_sigmas),
                                                                             np.array(dummy_sigmas))))
        # num_values_to_predict = 10
        # r_train_idx = randint(a=0, b=len(x_train) - num_values_to_predict)
        # print('pred train  =',
        #       print_np_arr(self.model.predict(x_train[r_train_idx:r_train_idx + num_values_to_predict]).flatten()))
        # print('truth train =', print_np_arr(y_train[r_train_idx:r_train_idx + num_values_to_predict].flatten()))
        # r_test_idx = randint(a=0, b=len(x_test) - num_values_to_predict)
        # print('pred  test  =',
        #       print_np_arr(self.model.predict(x_test[r_test_idx:r_test_idx + num_values_to_predict]).flatten()))
        # print('truth test  =', print_np_arr(y_test[r_test_idx:r_test_idx + num_values_to_predict].flatten()))
        # print('_' * 80)
        # print('\n')


m = Sequential()
m.add(LSTM(4, input_shape=(LSTM_WINDOW_SIZE, INPUT_SIZE)))
m.add(Dense(4, activation='sigmoid'))
m.add(Dense(1, activation='linear'))


# PAPER: with mean absolute percent error (MAPE) as the objective loss function
# PAPER: The model is trained by the 'Adam' method

def sigma_loss(y_true, y_pred):
    real_y_true = y_true * std + mean
    real_y_pred = y_pred * std + mean
    return K.mean(K.abs(real_y_true - real_y_pred) / real_y_true) * 100


m.compile(optimizer=Adam(lr=0.001), loss=sigma_loss)  # mape
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
