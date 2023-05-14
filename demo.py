from util import getdata
from TITR import TITR
import numpy as np

X_train, A_train, Y_train, optA_train = getdata(sample_size = 400, case = 3, seed = 100, outcome_shift=None)
X_test, A_test, Y_test, optA_test = getdata(sample_size = 10000, case = 3, seed = 400, outcome_shift=None)

model = TITR(kernel="rbf", gamma=0.0000001, sigma=0.1)

_ = model.fit(X_train, A_train, Y_train, max_iter=10)

print(model.beta0)
print(model.beta)

D = model.predict(X_test)

output = model.evaluate(Y_test, A_test, D, optA_test)

print("accuracy = {}".format(output[0]))
print("value = {}".format(output[1]))
