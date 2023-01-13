from util import getdata
from mcowl import MCOWL
import matplotlib.pyplot as plt
import numpy as np



Y, X, A, optA = getdata(500, case=4, seed=1)
Yt, Xt, At, optAt = getdata(500, case=4, seed=2)

A = 2 * A - 1
At = 2 * At - 1
optA = 2 * optA - 1
optAt = 2 * optAt - 1


Y = Y - np.min(Y) + 0.1
Yt = Yt - np.min(Yt) + 0.1

mcowl = MCOWL(layer=3, width=128, act="relu", loss="lsl")
history = mcowl.fit(Y, X, A, Yt, Xt, At, optAt, device='cpu', verbose=1, epochs=200, learning_rate=1e-2, batch_size=32)

D = mcowl.predict(Xt)
accuracy, value = mcowl.evaluate(Yt, At, D, optAt)

print("--- accuracy: {0} ---".format(accuracy))
print("--- value: {0} ---".format(value))

