import time

import tensorflow as tf
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_predict
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import confusion_matrix
from utils import two_folds_cv, save_scores_to_csv, file_to_lists, normalize_temps

plt.rcdefaults()

EXPERIMENT_NAME = "exp_comparison.csv"


def build_model_n(neurons_no):
    model = keras.Sequential([
        keras.layers.Dense(neurons_no, activation='relu', input_shape=[6]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    sgd = optimizers.sgd(lr=0.1, momentum=0.5)
    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def apply_2cv(neurons_number, xs, ys, iterations_no=5):
    """
    :param neurons_number:
    :param xs: input data, NxD np.array
    :param ys: output data, Nx1 np.array
    :param iterations_no: number of iterations of 2cv cycle (ex. 5 in case of 5x2cv)
    :return: table of scores
    """
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Hidden layers")
    header.append("Mean")
    scores_summary = [header]
    neurons_number = [5,10,15]
    for i in range(10):
        ts1 = time.time()
        scores = two_folds_cv(lambda: build_model_n(10), xs, ys, iterations_no)  # calculate scores in percent
        ts2 = time.time()
        print(ts2 - ts1)
        scores.append(np.mean(scores))
        scores.insert(0, 10)
        scores_summary.append(scores)
    return scores_summary


def save_scores(scores_summary):
    save_scores_to_csv(EXPERIMENT_NAME, scores_summary)



def proceed_experiment():
    data, xs, ys = file_to_lists()  # read data from the file
    normalize_temps(xs)  # normalize temperatures
    neurons_numbers = [5, 10, 15]

    scores_summary = apply_2cv(neurons_numbers, xs, ys)
    save_scores(scores_summary)


proceed_experiment()
