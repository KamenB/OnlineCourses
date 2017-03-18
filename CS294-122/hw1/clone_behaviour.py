#!/usr/bin/env python

import pickle
import numpy as np
import tf_util
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def normalize(train):
    mean, std = train.mean(), train.std()
    train = (train - mean) / std
    return train

def shuffle_datset(X, Y):
    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    Y = Y[perm, :]
    return X, Y

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def init_model(D, A):
    model = Sequential()
    model.add(Dense(128, input_dim=D, activation='relu'))
    model.add(Dense(A))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def split_dataset(X, Y, percent_train, percent_val):
    N = X.shape[0]
    trainEnd = int(N * percent_train / 100)
    valEnd = int(N * (percent_train + percent_val) / 100)
    trainX, trainY = X[:trainEnd, :], Y[:trainEnd, :]
    valX, valY = X[trainEnd:valEnd, :], Y[trainEnd:valEnd, :]
    testX, testY = X[valEnd:, :], Y[valEnd:, :]
    return trainX, trainY, valX, valY, testX, testY

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    args = parser.parse_args()

    # Dataset
    expert_data = pickle.load(open(args.expert_data_file, "rb"))
    X, Y = expert_data['observations'], expert_data['actions']

    # Shuffle dataset
    # X, Y = shuffle_datset(X, Y)

    # Normalize dataset
    # X = normalize(X)

    N, D = X.shape
    A = Y.shape[2]

    # Reshape Y from (N, 1, A) to (N, A)
    Y = Y.reshape(N, A)

    trainX, trainY, valX, valY, testX, testY = split_dataset(X, Y, 70, 30)

    model = init_model(D, A)

    history = model.fit(trainX, trainY, epochs=100, batch_size=100, verbose=2, validation_data=(valX, valY))
    # plot_history(history)

    clone_url = "clones/" + args.expert_data_file.split("/")[-1].split(".")[0] + ".h5"
    model.save(clone_url)

if __name__ == '__main__':
    main()
