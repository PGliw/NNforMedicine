import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier
from utils import two_folds_cv, save_scores_to_csv, file_to_lists, normalize_temps
import confusion_matrix
plt.rcdefaults()

EXPERIMENT_NAME = "exp_optimisation_alg.csv"


def build_model_with_optimizer(optimizer):
    model = keras.Sequential([
        keras.layers.Dense(5, activation='relu', input_shape=[6]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_learning_history(optimizers_with_names, x_train, y_train):
    """
    Shows how loss magnitude depends on epochs and neurons number
    :param optimizers_with_names: list of pairs (optimizer, optimizer_name)
    :param y_train: output train data Nx1
    :param x_train: input train data NxD
    :return:
    """
    colors = ['blue', 'green', 'red']
    for counter, optimizer_with_name in enumerate(optimizers_with_names):
        optimizer, optimizer_name = optimizer_with_name
        history = build_model_with_optimizer(optimizer).fit(x_train, y_train, epochs=500)
        plt.title("History of learning for different optimizers".format(optimizer_name))
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter % len(colors)],
                 label='{} optimizer'.format(optimizer_name))
    plt.legend()
    plt.show()


def apply_2cv(optimizers_with_names, xs, ys, iterations_no=5):
    """
    :param optimizers_with_names: list of pairs (optimizer, optimizer_name)
    :param xs: input data, NxD np.array
    :param ys: output data, Nx1 np.array
    :param iterations_no: number of iterations of 2cv cycle (ex. 5 in case of 5x2cv)
    :return: table of scores
    """
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Optimizer")
    header.append("Mean")
    scores_summary = [header]
    for optimizer, name in optimizers_with_names:
        scores = two_folds_cv(lambda: build_model_with_optimizer(optimizer), xs, ys, iterations_no, epochs=30)  # calculate scores in percent
        scores.append(np.mean(scores))
        scores.insert(0, name)
        scores_summary.append(scores)
    return scores_summary


def save_scores(scores_summary):
    save_scores_to_csv(EXPERIMENT_NAME, scores_summary)


def plot_confusion_matrices(optimizers_with_names, xs, ys):
    """
    :param optimizers_with_names: list of pairs (optimizer, optimizer_name)
    :param xs: xs: input data, NxD np.array
    :param ys: ys: output data, Nx1 np.array
    :return:
    """
    for optimizer, name in optimizers_with_names:
        estimator = KerasClassifier(build_fn=lambda: build_model_with_optimizer(optimizer), epochs=30, batch_size=5, verbose=0)
        y_pred = cross_val_predict(estimator, xs, ys, cv=5)
        confusion_matrix.plot_confusion_matrix(y_true=ys.astype(int), y_pred=y_pred.astype(int),
                                               classes=[0, 1, 2, 3], title="Confusion matrix tasted on {} samples for "
                                                                           "{} optimizer".format(
                len(y_pred), name))
        plt.show()


def proceed_experiment():
    data, xs, ys = file_to_lists()  # read data from the file
    normalize_temps(xs)  # normalize temperatures
    optimizers_with_names = [
        (keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False), "SGD - no momentum"),
        (keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False), "SGD - with momentum"),
        (keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), "ADAM")
        #   ('adam', "ADAM")
    ]
    plot_learning_history(optimizers_with_names, xs, ys)
    plot_confusion_matrices(optimizers_with_names, xs, ys)
    scores_summary = apply_2cv(optimizers_with_names, xs, ys)
    save_scores(scores_summary)


proceed_experiment()
