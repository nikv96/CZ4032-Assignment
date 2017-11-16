from helper import *
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
seed = 7
np.random.seed(seed)


def cost_rmspe(y_originals, y_predictions):
    return tf.sqrt(tf.reduce_mean((tf.square((y_predictions - y_originals) / y_predictions))))


filepath = "modelRNNVal.h5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min')
list_callbacks = [checkpoint]
print('Getting data ...')

x = np.concatenate(np.load('bigx.dill'))
y = np.concatenate(np.load('bigy.dill'))
print(x.shape[2])
in_neurons = x.shape[2]
num_hidden_nodes = 500
num_hidden_nodes_2 = 500
output_neuron = 1
num_epochs = 10
evaluation = []

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x, y):
    try:
        print('Loading simple DLSTM ...')
        model = load_model("modelRNN.h5", custom_objects={"cost_rmspe": cost_rmspe})
    except OSError as e:
        print('Creating simple DLSTM ...')
        model = Sequential()
        model.add(
            LSTM(
                num_hidden_nodes,
                input_dim=in_neurons,
                return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(output_neuron, input_dim=num_hidden_nodes))
        model.compile(loss=cost_rmspe, optimizer='rmsprop', metrics=['accuracy'])

    print('Fitting model ...')
    model.fit(
        x[train],
        y[train],
        batch_size=64,
        shuffle=True,
        epochs=10,
        verbose=1,
        callbacks=list_callbacks)
    score = model.evaluate(x[test], y[test], verbose=0)
    cvscores.append(score[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print('Done ...')
