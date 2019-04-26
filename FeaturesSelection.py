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
labels = []

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


'''
def feature_distribution(xs, feature_no, group_no):
    ys = kolmog.distribution_fun(xs)
    plt.subplot(121)
    plt.title("Feature {} in group {}".format(feature_no, group_no))
    plt.plot(xs, 'r*')
    plt.ylabel("Sample value")
    plt.xlabel("Sample no")
    plt.subplot(122)
    plt.plot(ys, 'g*')
    plt.ylabel("Distribution fun")
    plt.xlabel("Sample no")
    plt.show()


#print(groups[0])
a100 = results_np[:, 0]
a200 = results_np[:, 1]
a300 = results_np[:, 2]
a400 = results_np[:, 3]
a500 = results_np[:, 4]
a600 = results_np[:, 5]


feature_distribution(a100, 0, 0)
feature_distribution(a200, 1, 0)
feature_distribution(a300, 2, 0)
feature_distribution(a400, 3, 0)
feature_distribution(a500, 4, 0)


a1 = results_np[:,0]
a2 = results_np[:,1]
a3 = results_np[:,2]
a4 = results_np[:,3]
a5 = results_np[:,4]
a6 = results_np[:,5]
d1 = results_np[:,6]
d2 = results_np[:,7]
'''
# TODO divide results_np into submatricies according to one of 4 illnesses groups
