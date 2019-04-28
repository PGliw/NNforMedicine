import csv

def foo(a, b):
    return a * b


def foo_2(b):
    return foo(2, b)


lambdas = []

for i in range(5):
    lambdas.append(lambda b: foo(i, b))

for i in range(5):
    for b in range(5):
        print(lambdas[i](b))

for counter, value in enumerate(['c', 'b', 'a'], 10):
    print(counter, value)

header = ["{} score".format(i) for i in range(5)]
header.insert(0, "Neuron_no")
scores_conclusion = [header, [1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15]]
print(scores_conclusion)

def save_scores_to_csv(filename, scores_summary):
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter = ';')    # polish exel uses semicolons
        wr.writerows(scores_summary)


save_scores_to_csv("trial_res.csv", scores_conclusion)