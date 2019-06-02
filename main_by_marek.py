from __future__ import absolute_import, division, print_function

import time

import numpy as np
from scipy.stats import ks_2samp, stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import statistics

plt.rcdefaults()
import seaborn as sns  # for data visualisation

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

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

data_file = open("diagnosis01dots.csv", "r+")
print()

groups = [[], [], [], []]  # groups of results: 00, 01, 10, 11
results = []

features = []
feature_names = ["temperature", 'nausea', "lumber pain", "urine pushing", "micturition pains", "burning"]
labels = []
label_names = ("healthy", "nephritis", "inflammation", "nephritis and inflammation")

for line in data_file:
    no_nl_line = line.rstrip("\n\r")  # remove new line sign
    list_line = no_nl_line.split(";")  # split line into features
    result = [float(i) for i in list_line]  # save line values as floats
    result[0] = (result[0] - 35) / 6  # normalization of temperature
    results.append(result)  # add line to the results
    features.append(np.array(result[:(-2)]))  # features data

    d1, d2 = result[-2], result[-1]  # decisions
    label = 0
    if d1 == 0 and d2 == 0:
        label = 0  # healthy patient
    elif d1 == 0 and d2 == 1:
        label = 1  # only nephritis
    elif d1 == 1 and d2 == 0:
        label = 2  # only inflammation
    else:
        label = 3  # both nephritis and inflammation

    groups[label].append(result)
    labels.append(np.array(label))

features = np.array(features)
labels = np.array(labels)
# print(features)
# print("Labels:")
# print(labels)
# Split data into training and test sets
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

# labels_occurrence = [labels.count(i) for i in range(len(label_names))]
# y_pos = np.arange(len(label_names))

'''
# Plot diseases occurrence
plt.bar(y_pos, labels_occurrence, align='center', alpha=0.5)
plt.xticks(y_pos, label_names)
plt.ylabel('Occurrence')
plt.title('Occurrence of diseases')
plt.show()
'''


def plot_histogram(group_data, names_of_features, group_name, is_binary=True):
    """
    :param group_data: NxM array containing values of M features for N patients from a given patients group
    :param names_of_features: Mx1 array containing names for each feature given in group_data
    :param group_name: name (string) of group which results apply to - ex. 'healthy'
    :param is_binary: boolean indicating weather the feature is binary (0.0 or 1.0) or continuous (from 0.0 to 1.0)
    :return:
    """
    plt.style.use('seaborn-deep')
    if is_binary:
        plt.hist(group_data, bins=[-0.05, 0.05, 0.95, 1.05], label=names_of_features)
        plt.xticks([0, 1], ["no", "yes"])
    else:
        plt.hist(group_data, label=names_of_features)
        plt.xlim(0, 1)
    plt.title(group_name)
    plt.xlabel("Feature value")
    plt.ylabel('Number of patients')
    plt.legend(loc='upper right')
    plt.show()


"""
healthy, nephritis, inflammation, nep_and_inf = np.array(groups[0]), np.array(groups[1]), np.array(groups[2]), np.array(
    groups[3])
continuous_features_indexes, binary_features_indexes = [0], [1, 2, 3, 4, 5]
continuous_features_names, binary_features_names = feature_names[0], feature_names[1:6]
patient_groups = [healthy, nephritis, inflammation, nep_and_inf]
descriptions = [("Binary features values of a healthy patient (0=no, 1=yes)",
                 "Continuous features values of a healthy patient"),
                ("Binary features values of a patient with nephritis(0=no, 1=yes)",
                 "Continuous features values of a patient with nephritis"),
                ("Binary features values of a patient with inflammation (0=no, 1=yes)",
                 "Continuous features values of a patient with inflammation"),
                ("Binary features values of a patient with nephritis and inflammation(0=no, 1=yes)",
                 "Continuous features values of a patient with nephritis and inflammation")
                ]
for patient_group, (description_binary, description_continuous) in zip(patient_groups, descriptions):
    plot_histogram([patient_group[:, i] for i in binary_features_indexes],
                   binary_features_names, description_binary, is_binary=True)

    plot_histogram([patient_group[:, i] for i in continuous_features_indexes],
                   continuous_features_names, description_continuous, is_binary=False)
plot_histogram([healthy[:, i] for i in [1, 2, 3, 4, 5]], feature_names[1:6], "Binary features values of a healthy "
                                                                             "patient (0=no, 1=yes)")
plot_histogram(healthy[0], feature_names[0], "Continuous features values of a healthy "
                                             "patient", is_binary=False)

"""


## Plot feature histograms for each of 4 patients groups
## "healthy", "nephritis", "inflammation", "nephritis and inflammation"

def build_model_n(neurons_no):
    model = keras.Sequential([
        keras.layers.Dense(neurons_no, activation='relu', input_shape=[6]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


"""
global neurons_numbers, neurons_no
build_model_funs = []
for neurons_no in neurons_numbers:
    build_model_funs.append(lambda: build_model_n(neurons_no))
"""

neurons_numbers = [5, 10, 15]


def learning_history(neurons_numbers):
    """
    :param neurons_numbers: number of neurons in the hidden layer
    plot showing how loss magnitude depends on epochs and neurons number
    :return:
    """
    colors = ['blue', 'green', 'red']
    for counter, neurons_no in enumerate(neurons_numbers):
        history = build_model_n(neurons_no).fit(features, labels, epochs=500)
        plt.title("History of learning for {} neurons in the hidden layer".format(neurons_no))
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter % len(colors)],
                 label='{} hidden neurons'.format(neurons_no))

    plt.legend()
    plt.show()


# learning_history(neurons_numbers)


def two_folds_cv(build_model_fun, iterations_no=5):
    """
    :param build_model_fun: function that returns compiled Keras model
    :param iterations_no: number of iterations of cross validation
    :return: list of scores for each iteration
    """
    kfold = KFold(n_splits=2, shuffle=True)
    estimator = KerasClassifier(build_fn=build_model_fun, epochs=100, batch_size=5,
                                verbose=0)  # object implementing fit
    list_of_scores = []
    for i in range(iterations_no):
        scores = cross_val_score(estimator, features, labels, cv=kfold)
        list_of_scores.append(scores[0] * 100)  # change to percent

    print(list_of_scores)
    return list_of_scores


def apply_2cv(neurons_numbers, iterations_no=5):
    header = ["{} score".format(i) for i in range(5)]
    header.insert(0, "Neuron_no")
    header.append("Mean")
    scores_summary = [header]
    for neurons_no in neurons_numbers:
        scores = two_folds_cv(lambda: build_model_n(neurons_no), iterations_no)  # calculate scores in percent
        scores.append(np.mean(scores))
        scores.insert(0, neurons_no)
        scores_summary.append(scores)
    return scores_summary


# scores_summary = apply_2cv(neurons_numbers)


def save_scores_to_csv(filename, scores_summary):
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=';')  # polish exel uses semicolons
        wr.writerows(scores_summary)


# save_scores_to_csv("wyniki1.csv", scores_summary)

'''

def kolmogorov4nth_feature(number_of_feature1, number_of_feature2, number_of_class):
    feature1_data = []
    feature2_data = []
    for i in range(120):
        if labels[i] == number_of_class:
            feature1_data.append(features[i][number_of_feature1])
            feature2_data.append(features[i][number_of_feature2])
    return ks_2samp(feature1_data, feature2_data).__getitem__(0)

def kolmogorovElimination(treshold):
    dataSet = list(features)
    kolmogorov_results = [[[] for x in range(6)] for y in range(6)]
    for k in range(4):
        w = 6
        for j in range(w):
            for i in range(j, w):
                if j != i:
                    print('{}{}{}{}'.format("Kolmogorow for ", feature_names[i], " and ", feature_names[j]))
                    kolmogorov_results[j][i] = kolmogorov4nth_feature(i, j, k)
                    if kolmogorov_results[j][i] < treshold:
                        print(kolmogorov_results[j][i])
        filename = '{}{}{}'.format("Kolmogorov", k, ".csv")
        with open(filename, 'w') as myfile:
            wr = csv.writer(myfile, delimiter=';')
            wr.writerows(kolmogorov_results)
        print(['{:30}'.format(item) for item in feature_names])
        print(kolmogorov_results)
        # print('\n'.join([''.join(['{:30}'.format(item) for item in row]) for row in kolmogorov_results]))
        print("\n")


kolmogorovElimination(0.01)
'''


class RankingMember:
    """
    Class made to preserve association between feature name and value during sorting
    """

    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Feature: " + str(self.feature) + " with value: " + str(self.value) + "\n"


ranking = []


def kolmogorovFeatureRankingByMeanValue(features, label_names, feature_names):
    for featureIterator in range(6):
        results4singleFeature = []
        for classIterator in range(4):
            firstSet = []
            for i in range(120):
                if labels[i] == classIterator:
                    firstSet.append(features[i][featureIterator])
            for j in range(classIterator, 4):
                if classIterator != j:
                    secondSet = []
                    for i in range(120):
                        if labels[i] == j:
                            secondSet.append(features[i][featureIterator])
                    # print('{}{}{}{}{}{}'.format("Kolmogorow for ", label_names[classIterator], " and ", label_names[j]," for feature ",feature_names[featureIterator]))
                    results4singleFeature.append(ks_2samp(firstSet, secondSet).__getitem__(0))
        if len(results4singleFeature) != 0:
            ranking.append(RankingMember(feature_names[featureIterator], statistics.mean(results4singleFeature)))
            print('{}{}'.format("Results for ", feature_names[featureIterator]))
            print(results4singleFeature)
    ranking.sort(key=lambda x: x.value)
    ranking.reverse()


kolmogorovFeatureRankingByMeanValue(features, label_names, feature_names)


def best_features_extractor(number_of_features):
    print("{} number of features".format(number_of_features))
    features_to_return = []
    for x in features:
        f2append = []
        for counter in range(number_of_features):
            name_of_next_best_feature = ranking[counter].feature
            f2append.append(x[feature_names.index(name_of_next_best_feature)])
        features_to_return.append(f2append)
    return features_to_return


def two_folds_cv_f(build_model_fun, features_number, iterations_no):
    """
    :param build_model_fun: function that returns compiled Keras model
    :param iterations_no: number of iterations of cross validation
    :return: list of scores for each iteration
    """
    kfold = KFold(n_splits=2, shuffle=True, random_state=3)
    estimator = KerasClassifier(build_fn=build_model_fun, epochs=50, batch_size=5,
                                verbose=0)  # object implementing fit
    list_of_scores = []
    for i in range(iterations_no):
        features1 = np.array(best_features_extractor(features_number))
        scores = cross_val_score(estimator, features1, labels, cv=kfold)
        list_of_scores.append(scores[0] * 100)  # change to percent
    print(list_of_scores)
    return list_of_scores


def build_model_nn(neurons_no, feature_no):
    model = keras.Sequential([
        keras.layers.Dense(neurons_no, activation='relu', input_shape=[feature_no]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def method5x2cv(neurons_number, features_number):
    ts1 = time.time()
    scores = two_folds_cv_f(lambda: build_model_nn(neurons_number, features_number), features_number, iterations_no=5)
    ts2 = time.time()
    scores.append(np.mean(scores))
    scores.insert(0, features_number)
    scores_summary = scores
    scores_summary.append(ts2 - ts1)
    return scores_summary


def learning_history_with_feature_ranking(neurons_numbers):
    """
    :param neurons_numbers: number of neurons in the hidden layer
    plot showing how loss magnitude depends on epochs and neurons number
    :return:
    """
    colors = ['blue', 'green', 'red', 'yellow', 'black', 'magenta']
    header = ["{} score".format(i) for i in range(1, 6)]
    header.insert(0, "Number of features")
    header.append("Mean")
    header.append("Time 5x2cv")
    scores = [header]
    for counter in range(1, 7):
        print("{} counter value".format(counter))
        # features1 = best_features_extractor(counter)
        scores.append(method5x2cv(neurons_numbers, counter))
        """
        history = build_model_nn(neurons_numbers, counter).fit(np.array(features1), np.array(labels), epochs=500)
        plt.title("History of learning for different number of features as input")
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter - 1 % len(colors)],
                 label='{} of best features'.format(counter))
        
    plt.legend()
    plt.show()"""
    save_scores_to_csv("wyniki_dla_roznej_liczby_cech.csv", scores)


learning_history_with_feature_ranking(15)


