from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import random

'''
a1	Temperature of patient { 35C-42C }	
a2	Occurrence of nausea { yes, no }	
a3	Lumbar pain { yes, no }	
a4	Urine pushing (continuous need for urination) { yes, no }	
a5	Micturition pains { yes, no }	
a6	Burning of urethra, itch, swelling of urethra outlet { yes, no }	
d1	decision: Inflammation of urinary bladder { yes, no }	
d2	decision: Nephritis of renal pelvis origin { yes, no }
'''

groups = [[], [], [], []]  # groups of results: 00, 01, 10, 11
results = []

features = []
feature_names = ["temperature", 'nausea', "lumber pain", "urine pushing", "micturition pains", "burning"]
labels = []
label_names = ("healthy", "nephritis", "inflammation", "nephritis and inflammation")


def file_to_lists():
    """"
    :return: tuple: data - list of data from file, xs - list of feature values, ys - list of labels
    """
    data, xs, ys = [], [], []
    data_file = open("diagnosis01dots.csv", "r+")
    for line in data_file:
        no_nl_line = line.rstrip("\n\r")  # remove new line sign
        list_line = no_nl_line.split(";")  # split line into features
        data_line = [float(i) for i in list_line]  # save line values as floats
        data.append(data_line)  # add line to the results
        xs.append(data_line[:(-2)])
        d1, d2 = data_line[-2], data_line[-1]  # decisions
        if d1 == 0 and d2 == 0:
            label = 0  # healthy patient
        elif d1 == 0 and d2 == 1:
            label = 1  # only nephritis
        elif d1 == 1 and d2 == 0:
            label = 2  # only inflammation
        else:
            label = 3  # both nephritis and inflammation
        ys.append(label)
    return np.array(data), np.array(xs), np.array(ys)

def normalize_temps(xs):
    """
    Normalizes input temperature xs[:,0]
    :param xs: NxD np.array
    :return: Nothing
    """
    temperatures = xs[:, 0]
    minimal_temp = np.amin(temperatures)
    maximal_temp = np.amax(temperatures)
    delta = maximal_temp - minimal_temp
    normalized_temps = (temperatures - minimal_temp) / delta
    xs[:, 0] = normalized_temps


def data_to_groups(data):
    """"
    :param: data to divide to groups
    :return list of 4 lists of data - each for one patient group
    """
    groups = [[], [], [], []]
    for data_line in data:
        d1, d2 = data[-2], data[-1]  # decisions
        if d1 == 0 and d2 == 0:
            label = 0  # healthy patient
        elif d1 == 0 and d2 == 1:
            label = 1  # only nephritis
        elif d1 == 1 and d2 == 0:
            label = 2  # only inflammation
        else:
            label = 3  # both nephritis and inflammation
        groups[label].append(data_line)
    return groups


def plot_diseases_occurrence(ys):
    labels, labels_occurrence = np.unique(ys, return_counts=True)
    y_pos = np.arange(len(label_names))
    # Plot diseases occurrence
    plt.bar(y_pos, labels_occurrence, align='center', alpha=0.5)
    plt.xticks(y_pos, label_names)
    plt.ylabel('Occurrence')
    plt.title('Occurrence of diseases')
    plt.show()


def two_folds_cv(build_model_fun, xs, ys, iterations_no=5, epochs=60):
    """
    :param build_model_fun: function that returns compiled Keras model
    :param xs: NxD np.array of input data
    :param ys: Nx1 np.array of true output data
    :param iterations_no: number of iterations of cross validation
    :return: list of scores for each iteration
    """
    seed = 3
    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)  # two folds
    estimator = KerasClassifier(build_fn=build_model_fun, epochs=epochs, batch_size=5, verbose=0)  # object implementing fit
    list_of_scores = []
    for i in range(iterations_no):
        scores = cross_val_score(estimator, xs, ys, cv=kfold)
        print(scores)
        list_of_scores.append(scores[0]*100)    # change to percent
    return list_of_scores


def save_scores_to_csv(filename, scores_summary):
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=';')  # polish exel uses semicolons
        wr.writerows(scores_summary)


def divide_set_to_random_parts(xs, ys):
    xs_and_ys = list(zip(xs, ys))
    random.shuffle(xs_and_ys)
    xs_shuffled, y_shuffled = zip(*xs_and_ys)
    return list(xs_shuffled), list(y_shuffled)

