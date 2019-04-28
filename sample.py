def foo(a, b):
    return a*b

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