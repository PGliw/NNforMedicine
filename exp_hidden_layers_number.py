import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier
from utils import two_folds_cv, save_scores_to_csv, file_to_lists, normalize_temps
import confusion_matrix
plt.rcdefaults()

EXPERIMENT_NAME = "exp_layers_number.csv"


def build_model_with_hidden_layers_no(hidden_layers_no):
    if hidden_layers_no == 1:   # 1 hidden layer
        model = keras.Sequential([
            keras.layers.Dense(5, activation='relu', input_shape=[6]),
            keras.layers.Dense(4, activation=tf.nn.softmax)
        ])
    elif hidden_layers_no == 2:     # 2 hidden layers
        model = keras.Sequential([
            keras.layers.Dense(5, activation='relu', input_shape=[6]),
            keras.layers.Dense(5, activation='relu', input_shape=[10]),
            keras.layers.Dense(4, activation=tf.nn.softmax)
        ])
    else:   # 3 hidden layers
        model = keras.Sequential([
            keras.layers.Dense(5, activation='relu', input_shape=[6]),
            keras.layers.Dense(5, activation='relu', input_shape=[10]),
            keras.layers.Dense(5, activation='relu', input_shape=[10]),
            keras.layers.Dense(4, activation=tf.nn.softmax)
        ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_learning_history(hidden_layers_numbers, x_train, y_train):
    """
    Shows how loss magnitude depends on epochs and neurons number
    :param hidden_layers_numbers: hidden layers numbers list - elem. min.1, max. 3
    :param y_train: output train data Nx1
    :param x_train: input train data NxD
    :return:
    """
    colors = ['blue', 'green', 'red']
    for counter, hidden_layers_no in enumerate(hidden_layers_numbers):
        history = build_model_with_hidden_layers_no(hidden_layers_numbers).fit(x_train, y_train, epochs=500)
        plt.title("Learning history for different hidden layers number")
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter % len(colors)],
                 label='{} hidden layers'.format(hidden_layers_no))
    plt.legend()
    plt.show()


def apply_2cv(hidden_layers_numbers, xs, ys, iterations_no=5):
    """
    :param hidden_layers_numbers: hidden layers numbers list - elem. min.1, max. 3
    :param xs: input data, NxD np.array
    :param ys: output data, Nx1 np.array
    :param iterations_no: number of iterations of 2cv cycle (ex. 5 in case of 5x2cv)
    :return: table of scores
    """
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Hidden layers")
    header.append("Mean")
    scores_summary = [header]
    for neurons_no in hidden_layers_numbers:
        scores = two_folds_cv(lambda: build_model_with_hidden_layers_no(hidden_layers_numbers), xs, ys, iterations_no)  # calculate scores in percent
        scores.append(np.mean(scores))
        scores.insert(0, neurons_no)
        scores_summary.append(scores)
    return scores_summary


def save_scores(scores_summary):
    save_scores_to_csv(EXPERIMENT_NAME, scores_summary)


def plot_confusion_matrices(hidden_layers_numbers, xs, ys):
    """
    :param hidden_layers_numbers: hidden layers numbers list - elem. min.1, max. 3
    :param xs: xs: input data, NxD np.array
    :param ys: ys: output data, Nx1 np.array
    :return:
    """
    for hidden_layers_no in hidden_layers_numbers:
        estimator = KerasClassifier(build_fn=lambda: build_model_with_hidden_layers_no(hidden_layers_no), epochs=30, batch_size=5, verbose=0)
        y_pred = cross_val_predict(estimator, xs, ys, cv=5)
        confusion_matrix.plot_confusion_matrix(y_true=ys.astype(int), y_pred=y_pred.astype(int),
                                               classes=[0, 1, 2, 3], title="Confusion matrix tasted on {} samples for "
                                                                           "{} hidden layers".format(
                len(y_pred), hidden_layers_no))
        plt.show()


def proceed_experiment():
    data, xs, ys = file_to_lists()  # read data from the file
    normalize_temps(xs)  # normalize temperatures
    hidden_layers_numbers = [1, 2, 3]
    plot_learning_history(hidden_layers_numbers, xs, ys)
    plot_confusion_matrices(hidden_layers_numbers, xs, ys)
    scores_summary = apply_2cv(hidden_layers_numbers, xs, ys)
    save_scores(scores_summary)


proceed_experiment()
