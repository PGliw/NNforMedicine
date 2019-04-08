import numpy as np

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

#results = [[], [], [], []]  # 4 classes of diseases
results = []

for line in data_file:
    #print(line)
    result = [0.0, False, False, False, False, False, False, False]
    no_nl_line = line.rstrip("\n\r")
    list_line = no_nl_line.split(";")
    result = [float(i) for i in list_line]
    results.append(result)
    #print(result)

# print(results)
results_np = np.array(results)
#print(results_np)
a1 = results_np[:,0]
a2 = results_np[:,1]
a3 = results_np[:,2]
a4 = results_np[:,3]
a5 = results_np[:,4]
a6 = results_np[:,5]
d1 = results_np[:,6]
d2 = results_np[:,7]

# TODO divide results_np into submatricies according to one of 4 illnesses groups


