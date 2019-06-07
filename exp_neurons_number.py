import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier
from utils import two_folds_cv, save_scores_to_csv, file_to_lists, normalize_temps
import confusion_matrix
plt.rcdefaults()

EXPERIMENT_NAME = "exp_neurons_number.csv"


def build_model_n(neurons_no):
    model = keras.Sequential([
        keras.layers.Dense(neurons_no, activation='relu', input_shape=[6]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_learning_history(neurons_numbers, x_train, y_train):
    """
    Shows how loss magnitude depends on epochs and neurons number
    :param y_train: output train data Nx1
    :param x_train: input train data NxD
    :param neurons_numbers: number of neurons in the hidden layer
    :return:
    """
    colors = ['blue', 'green', 'red']
    for counter, neurons_no in enumerate(neurons_numbers):
        history = build_model_n(neurons_no).fit(x_train, y_train, epochs=500)
        plt.title("History of learning for {} neurons in the hidden layer".format(neurons_no))
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter % len(colors)],
                 label='{} hidden neurons'.format(neurons_no))
    plt.legend()
    plt.show()


def apply_2cv(neurons_numbers, xs, ys, iterations_no=5):
    """
    :param neurons_numbers: list of neurons numbers in hidden layer to test
    :param xs: input data, NxD np.array
    :param ys: output data, Nx1 np.array
    :param iterations_no: number of iterations of 2cv cycle (ex. 5 in case of 5x2cv)
    :return: table of scores
    """
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Neuron_no")
    header.append("Mean")
    scores_summary = [header]
    for neurons_no in neurons_numbers:
        scores = two_folds_cv(lambda: build_model_n(neurons_no), xs, ys, iterations_no)  # calculate scores in percent
        scores.append(np.mean(scores))
        scores.insert(0, neurons_no)
        scores_summary.append(scores)
    return scores_summary


def save_scores(scores_summary):
    save_scores_to_csv(EXPERIMENT_NAME, scores_summary)


def plot_confusion_matrices(neurons_numbers, xs, ys):
    """
    :param neurons_numbers: neurons_numbers: list of neurons numbers in hidden layer to test
    :param xs: xs: input data, NxD np.array
    :param ys: ys: output data, Nx1 np.array
    :return:
    """
    for neurons_no in neurons_numbers:
        estimator = KerasClassifier(build_fn=lambda: build_model_n(neurons_no), epochs=60, batch_size=5, verbose=0)
        y_pred = cross_val_predict(estimator, xs, ys, cv=5)
        confusion_matrix.plot_confusion_matrix(y_true=ys.astype(int), y_pred=y_pred.astype(int),
                                               classes=[0, 1, 2, 3], title="Confusion matrix tasted on {} samples for "
                                                                           "{} neurons in hidden layer".format(
                len(y_pred), neurons_no))
        plt.show()


def proceed_experiment():
    data, xs, ys = file_to_lists()  # read data from the file
    normalize_temps(xs)  # normalize temperatures
    neurons_numbers = [5, 10, 15]
    plot_learning_history(neurons_numbers, xs, ys)
    plot_confusion_matrices(neurons_numbers, xs, ys)
    scores_summary = apply_2cv(neurons_numbers, xs, ys)
    save_scores(scores_summary)


proceed_experiment()
