import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from utils import two_folds_cv, save_scores_to_csv, file_to_lists, normalize_temps
import confusion_matrix
plt.rcdefaults()

EXPERIMENT_NAME = "exp_neurons_number"


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
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Neuron_no")
    header.append("Mean")
    scores_summary = [header]
    for neurons_no in neurons_numbers:
        scores = two_folds_cv(lambda: build_model_n(neurons_no),xs, ys, iterations_no)  # calculate scores in percent
        scores.append(np.mean(scores))
        scores.insert(0, neurons_no)
        scores_summary.append(scores)
    return scores_summary


def save_scores(scores_summary):
    save_scores_to_csv(EXPERIMENT_NAME, scores_summary)


def proceed_experiment():
    data, xs, ys = file_to_lists()  # read data from the file
    normalize_temps(xs)  # normalize temperatures
    neurons_numbers = [5, 10, 15]
    plot_learning_history(neurons_numbers, xs, ys)
    x_train, x_test, y_train, y_test = train_test_split(xs,
                                                        ys,
                                                        test_size=0.33,
                                                        random_state=42)
    scores_summary = apply_2cv(neurons_numbers, x_train, y_train)
    save_scores(scores_summary)


proceed_experiment()

'''
# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(xs,
                                                    ys,
                                                    test_size=0.33,
                                                    random_state=42)
'''
