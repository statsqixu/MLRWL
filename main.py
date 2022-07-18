from util import getdata
from mcowl import MCOWL
import matplotlib.pyplot as plt
import numpy as np

def demo():

    Y, X, A, optA = getdata(200, case=1, seed=1)

    mcowl = MCOWL()
    history = mcowl.fit(Y, X, A, device='cpu', verbose=1, epochs=100, learning_rate=1e-4)

    D = mcowl.predict(X)
    accuracy, value = mcowl.evaluate(Y, A, D, optA)

    print("--- accuracy: {0} ---".format(accuracy))
    print("--- value: {0} ---".format(value))


if __name__ == "__main__":

    demo()