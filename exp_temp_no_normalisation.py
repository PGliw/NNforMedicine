import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier
from utils import two_folds_cv, save_scores_to_csv, file_to_lists, normalize_temps
import confusion_matrix

plt.rcdefaults()

EXPERIMENT_NAME = "exp_no_normalistaion.csv"


def build_model_with_activation_funs(activation_fun_1, acttivation_fun_2):
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=[6]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_learning_history(activation_funs_with_names_list, x_train, y_train):
    """
    Shows how loss magnitude depends on epochs and neurons number
    :param activation_funs_with_names_list: list of pairs ((act_fun_1, act_fun_2), (name_1, name_2))
    :param y_train: output train data Nx1
    :param x_train: input train data NxD
    :return:
    """
    colors = ['blue', 'green', 'red']
    for counter, activation_funs_with_names in enumerate(activation_funs_with_names_list):
        activation_funs, activation_names = activation_funs_with_names
        history = build_model_with_activation_funs(activation_funs[0], activation_funs[1]).fit(x_train, y_train,
                                                                                               epochs=500)
        plt.title("History of learning for activation functions")
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter % len(colors)],
                 label='{} fun for hidden layer, {} fun for output layer'.format(activation_names[0],
                                                                                 activation_names[1]))
    plt.legend()
    plt.show()


def apply_2cv(activation_funs_with_names_list, xs, ys, iterations_no=5):
    """
    :param activation_funs_with_names_list: list of pairs ((act_fun_1, act_fun_2), (name_1, name_2))
    :param xs: input data, NxD np.array
    :param ys: output data, Nx1 np.array
    :param iterations_no: number of iterations of 2cv cycle (ex. 5 in case of 5x2cv)
    :return: table of scores
    """
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Activation functions")
    header.append("Mean")
    scores_summary = [header]
    for activation_funs, activation_names in activation_funs_with_names_list:
        scores = two_folds_cv(lambda: build_model_with_activation_funs(activation_funs[0], activation_funs[1]), xs, ys,
                              iterations_no, epochs=30)  # calculate scores in percent
        scores.append(np.mean(scores))
        scores.insert(0, activation_names)
        scores_summary.append(scores)
    return scores_summary


def save_scores(scores_summary):
    save_scores_to_csv(EXPERIMENT_NAME, scores_summary)


def plot_confusion_matrices(activation_funs_with_names_list, xs, ys):
    """
    :param activation_funs_with_names_list: list of pairs ((act_fun_1, act_fun_2), (name_1, name_2))
    :param xs: xs: input data, NxD np.array
    :param ys: ys: output data, Nx1 np.array
    :return:
    """
    for activation_funs, activation_names in activation_funs_with_names_list:
        estimator = KerasClassifier(
            build_fn=lambda: build_model_with_activation_funs(activation_funs[0], activation_funs[1]), epochs=30,
            batch_size=5, verbose=0)
        y_pred = cross_val_predict(estimator, xs, ys, cv=5)
        confusion_matrix.plot_confusion_matrix(y_true=ys.astype(int), y_pred=y_pred.astype(int),
                                               classes=[0, 1, 2, 3], title="Confusion matrix tasted on {} samples for "
                                                                           "{} and {} activation functions".format(
                len(y_pred), activation_names[0], activation_names[1]))
        plt.show()


def proceed_experiment():
    data, xs, ys = file_to_lists()  # read data from the file
    #   normalize_temps(xs)  # normalize temperatures
    activation_funs_with_names_list = [
        (('relu', tf.nn.softmax), ("Relu", "Softmax")),
        (('sigmoid', tf.nn.softmax), ("Sigmoid", "Softmax")),
        (('linear', tf.nn.softmax), ("Linear", "Softmax"))
    ]
    plot_learning_history(activation_funs_with_names_list, xs, ys)
    plot_confusion_matrices(activation_funs_with_names_list, xs, ys)
    scores_summary = apply_2cv(activation_funs_with_names_list, xs, ys)
    save_scores(scores_summary)


proceed_experiment()
