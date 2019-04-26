import numpy as np
import Kolmogorov as kolmog
import matplotlib.pyplot as plt

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
feature_names = ["temperature", "lumber pain", "urine pushing", "micturition pains", "burning"]
labels = []
label_names = ["healthy", "nephritis", "inflammation", "nephritis and inflammation"]

for line in data_file:
    no_nl_line = line.rstrip("\n\r")  # remove new line sign
    list_line = no_nl_line.split(";")  # split line into features
    result = [float(i) for i in list_line]  # save line values as floats

    results.append(result)  # add line to the results
    features.append(result[:(-2)])  # features data

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
    labels.append(label)


results_np = np.array(results)  # list to np.array
print(results_np)



