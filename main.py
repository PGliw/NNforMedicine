from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.model_selection import train_test_split
import Kolmogorov as kolmog
import matplotlib.pyplot as plt
import csv

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


def build_model_n(neurons_no):
    model = keras.Sequential([
        keras.layers.Dense(neurons_no, activation='relu', input_shape=[6]),  # eg. 5, 10, 15 neurons
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


neurons_numbers = [5, 10, 15]
build_model_funs = []
for neurons_no in neurons_numbers:
    build_model_funs.append(lambda: build_model_n(neurons_no))


def learning_history(neurons_numbers):
    """
    :param neurons_numbers: number of neurons in the hidden layer
    plot showing how loss magnitude depends on epochs and neurons number
    :return:
    """
    colors = ['blue', 'green', 'red']
    for counter, neurons_no in enumerate(neurons_numbers):
        history = build_model_n(neurons_no).fit(train, train_labels, epochs=500)
        plt.title("History of learning for {} neurons in the hidden layer".format(neurons_no))
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'], color=colors[counter % len(colors)],
                 label='{} hidden neurons'.format(neurons_no))

    plt.legend()
    plt.show()


#   learning_history(neurons_numbers)


def two_folds_cv(build_model_fun, iterations_no=5):
    """
    :param build_model_fun: function that returns compiled Keras model
    :param iterations_no: number of iterations of cross validation
    :return: list of scores for each iteration
    """
    seed = 3
    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
    estimator = KerasClassifier(build_fn=build_model_fun, epochs=60, batch_size=5, verbose=0)  # object implementing fit
    list_of_scores = []
    for i in range(iterations_no):
        scores = cross_val_score(estimator, train, train_labels, cv=kfold)
        list_of_scores.append(scores[0]*100)    # change to percent

    #   print(list_of_scores)
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


scores_summary = apply_2cv(neurons_numbers)
print(scores_summary)


def save_scores_to_csv(filename, scores_summary):
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=';')  # polish exel uses semicolons
        wr.writerows(scores_summary)

save_scores_to_csv("wyniki.csv", scores_summary)
