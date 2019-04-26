from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.model_selection import train_test_split
import Kolmogorov as kolmog
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns # for data visualisation

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
    result[0] = (result[0] - 35) / 6    # normalization of temperature
    results.append(result)  # add line to the results
    features.append(np.array(result[:(-2)]))  # features data

    d1, d2 = result[-2], result[-1]  # decisions
    label = 0
    if d1 == 0 and d2 == 0:
        label = 0   # healthy patient
    elif d1 == 0 and d2 == 1:
        label = 1   # only nephritis
    elif d1 == 1 and d2 == 0:
        label = 2   # only inflammation
    else:
        label = 3   # both nephritis and inflammation

    groups[label].append(result)
    labels.append(np.array(label))

features = np.array(features)
labels = np.array(labels)


# Split data into training and test sets
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

#labels_occurrence = [labels.count(i) for i in range(len(label_names))]
#y_pos = np.arange(len(label_names))

'''
# Plot diseases occurrence
plt.bar(y_pos, labels_occurrence, align='center', alpha=0.5)
plt.xticks(y_pos, label_names)
plt.ylabel('Occurrence')
plt.title('Occurrence of diseases')
plt.show()
'''
#   TODO normalize temperatures

#   Build the NN model
def bulid_model():
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(6,)),
    #    keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#   model.fit(train, train_labels, epochs=30)

#   print("Evaluation")
#   model.evaluate(test, test_labels)

# 2 folds cross validation
seed = 3
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
estimator = KerasClassifier(build_fn=bulid_model, epochs=60, batch_size=5, verbose=0)   # object implementing fit
list_of_scores = []
for i in range(5):
    scores = cross_val_score(estimator, train, train_labels, cv=kfold)
    list_of_scores.append(scores)

print(list_of_scores)
#print("Baseline: %.2f%% (%.2f%%)" % (evals.mean()*100, evals.std()*100))
# TODO repeat 2cv 5 times