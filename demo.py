from util import *
from TITR import TITR
import numpy as np
from tqdm import tqdm

X_train, A_train, Y_train, optA_train = getdata(sample_size = 800, case = 3, seed = 0, outcome_shift=None, randomized=True)
X_test, A_test, Y_test, optA_test = getdata(sample_size = 10000, case = 3, seed = 200, outcome_shift=None, randomized=True)

model = TITR(kernel="linear", gamma=1)

_ = model.fit(X_train, A_train, Y_train, max_iter=100, verbose=1)

D = model.predict(X_test)

output = model.evaluate(Y_test, A_test, D, optA_test)

print("accuracy = {}".format(output[0]))
print("value = {}".format(output[1]))
