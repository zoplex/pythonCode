"""Softmax."""

scores = [3.0, 1.0, 0.2] # for 1 sample, 3 output classes

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    # e_x = np.exp(x - np.max(x))
    # retv = e_x / e_x.sum()

    retv = np.exp(x) / np.sum( np.exp(x), axis=0)
    return retv


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.5)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

# small and large numbers
n1 = 1000000000
for i in range(1,1000000):
    n1 = n1 + 1e-6

print(n1-1000000000)



