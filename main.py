from util import getdata
from mcowl import MCOWL
import matplotlib.pyplot as plt
import numpy as np

# def demo():

Y, X, A, optA = getdata(200, case=2, seed=1)

A = 2 * A - 1
optA = 2 * optA - 1

Y = Y - np.min(Y) + 0.1

mcowl = MCOWL(layer=5, width=50, act="relu", loss="ghl")
history = mcowl.fit(Y, X, A, device='cpu', verbose=1, epochs=500, learning_rate=1e-4, batch_size=16)

D = mcowl.predict(X)
accuracy, value = mcowl.evaluate(Y, A, D, optA)

print("--- accuracy: {0} ---".format(accuracy))
print("--- value: {0} ---".format(value))


# if __name__ == "__main__":

#     demo()