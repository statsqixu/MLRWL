from util import getdata
from TITR import TITR
import numpy as np

X_train, A_train, Y_train, optA_train = getdata(sample_size = 800, case = 1, seed = 0, outcome_shift=None)
X_test, A_test, Y_test, optA_test = getdata(sample_size = 10000, case = 1, seed = 200, outcome_shift=None)

model = TITR(kernel="linear", gamma=1)

_ = model.fit(X_train, A_train, Y_train, max_iter=100)

D = model.predict(X_test)

output = model.evaluate(Y_test, A_test, D, optA_test)

print("accuracy = {}".format(output[0]))
print("value = {}".format(output[1]))
