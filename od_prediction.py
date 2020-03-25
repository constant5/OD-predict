import random
import xml.etree.ElementTree as ET
from math import sqrt

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import sparse
import keras
from keras import backend as K
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Reshape, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU


flows = pd.read_csv('flows.csv')
trips = pd.read_csv('trips.csv')

# cut trips data
trips = trips[trips['sim_id'].isin(flows['sim_id'].unique())]

trips['cell_from'] =  trips['x_from'].astype(str).str.cat(trips['y_from'].astype(str), sep="_")
trips['cell_to'] =  trips['x_to'].astype(str).str.cat(trips['y_to'].astype(str), sep="_")


def flows_to_tensor(flows):
  """
  Transforms pandas dataframe to suitable for network format
  """
  coords_columns = ["sim_id", "x", "y", "time"]

  return sparse.COO(
      np.transpose(flows[coords_columns].values),
      flows['vehicles_number'],
      shape=(flows['sim_id'].max() + 1, 32, 32, flows['time'].max()+1)
  )


le_trips = preprocessing.LabelEncoder()


def trips_to_tensor(data):
  # only_origins = data[['x_from', 'y_from', 'x_to', 'y_to']].drop_duplicates().sort_values(by=['x_from', 'y_from', 'x_to', 'y_to'])

  data['cell_from'] =  data['x_from'].astype(str).str.cat(data['y_from'].astype(str), sep="_")
  data['cell_to'] =  data['x_to'].astype(str).str.cat(data['y_to'].astype(str), sep="_")
  data['code'] = data['cell_from'].astype(str).str.cat(data['cell_to'].astype(str), sep="_")
  trips_hist = data.groupby('code')['sim_id'].count()
  threshold = trips_hist.quantile(0.95)
  trips_interested = np.array(trips_hist[trips_hist > threshold].index)
  data = data[data['code'].isin(trips_interested)]
  data['code_encoded'] = le_trips.fit_transform(data['code'])
  trips_per_sim = data.groupby(['sim_id', 'code_encoded']).size().reset_index()
  return sparse.COO(
      np.transpose(trips_per_sim[['sim_id', 'code_encoded']].values),
      trips_per_sim[0].values,
      shape=((data['sim_id'].max()+1, data['code_encoded'].max()+1))
  )


output = trips_to_tensor(trips)

matrix_size = 32
iterations_number = trips['time'].values.max()+1


def make_final_model():
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(3,3), padding="same"))
    model.add(Activation(LeakyReLU(alpha=0.1)))
    model.add(Conv2D(filters=8, kernel_size=(3,3), padding="same"))
    model.add(Activation(LeakyReLU(alpha=0.1)))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same"))
    model.add(Activation(LeakyReLU(alpha=0.1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(Activation(LeakyReLU(alpha=0.1)))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    # model.add(Dense(2*graph_size**2))
    model.add(Dense(output.shape[1]))
    model.add(Dropout(0.5))
    model.add(Activation(LeakyReLU(alpha=0.1)))

    input_shape = (None, matrix_size, matrix_size, iterations_number)

    model.build(input_shape)
    return model

X = flows_to_tensor(flows)
Y = output

split_val = round(trips['sim_id'].max() * 0.6)
split_test = round(trips['sim_id'].max() * 0.9)

data_train = X[0:split_val,:,:,:]
labels_train = Y[0:split_val,:]
data_val = X[split_val:split_test,:,:,:]
labels_val = X[split_val:split_test,:,:]
data_test = X[split_test:,:,:]
labels_test = Y[split_test:,:]


INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 215
EPOCHS = 3

# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
model_final = make_final_model()  # define our model

# prepare model for fitting (loss, optimizer, etc)
model_final.compile(
    loss='mse',
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['mse']  # report accuracy during training
)


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        # clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


plot_losses = PlotLosses()


# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch


# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model_final.optimizer.lr))


# fit model
train_history = model_final.fit(
    data_train, labels_train,  # prepared data
    # steps_per_epoch=BATCH_SIZE,
    # validation_steps=BATCH_SIZE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
               keras.callbacks.LearningRateScheduler(lr_scheduler),
               LrHistory(),
               plot_losses
              ],
    validation_data=(data_val, labels_val),
    shuffle=True,
    verbose=0,
    initial_epoch=0
)

model_final.save_weights('model_final.h5')

trips_predicted = model_final.predict(data_test)

print("Quality on the test set:", sqrt(mean_squared_error(trips_predicted, labels_test)))


def tensor_to_trips(data):
    """
    Transforms numpy array with predictions to a convinient pandas dataframe

    Args:
      data: numpy array with predictions
    Returns:
      dataframe with the header: [sim_id, x_from, y_from, x_to, y_to]
    """

    dataset = []
    codes = le_trips.inverse_transform(range(data.shape[1]))
    for sim_id in range(data.shape[0]):
        for cell in range(data.shape[1]):
            code = codes[cell]
            x_from, y_from, x_to, y_to = code.split('_')
            number = int(data[sim_id, cell])
            for i in range(number):
                dataset.append([sim_id, x_from, y_from, x_to, y_to])
    return pd.DataFrame(dataset, columns=['sim_id', 'x_from', 'y_from', 'x_to', 'y_to'])

tensor_to_trips(trips_predicted).to_csv('trips_predicted.csv')