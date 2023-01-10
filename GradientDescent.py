import numpy as np

theta = 1e-3
def min(x):
    return (np.exp(x) - 2 / np.exp(x))**2

def df(x):
    return 2*(np.exp(x) + 2 / np.exp(x))*(np.exp(x) - 2 / np.exp(x))

def gradient_descent(learning_rate, x0, n_iterations = 100):
    x = [x0]
    for _ in range(n_iterations):
        x_new = x[-1] - learning_rate*df(x[-1])
        if abs(df(x_new)) < 1e-3:
            continue
        x.append(x_new)
    return x

x = gradient_descent(0.01, 2)
print('The min of f(x) is {0} at x = {1}'.format(min(x[-1]), x[-1]))