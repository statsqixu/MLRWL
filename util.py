import numpy as np

def _categorical_treatment(A):
    
    """
    Create categorical representation of multi-channel treatment
    """

    _, A_cate = np.unique(A, return_inverse=True, axis=0)

    return A_cate

# generate data based on linear decision boundary

def getdata(sample_size, case=1, outcome_shift=None, 
            seed = None):

    if seed is not None:

        np.random.seed(seed)

    X = np.random.uniform(-1, 1, (sample_size, 10))

    m = 1 + X[:, 0] + 2 * X[:, 1]

    if case == 1:
        
        A = np.random.choice([-1, 1], (sample_size, 2))
        A_cate = _categorical_treatment(A)

        trt_panel = np.zeros((sample_size, 4))
        trt_panel[:, 0] = 0 * (X[:, 0] + X[:, 1] < 0) * (-X[:, 0] + X[:, 1] < 0)
        trt_panel[:, 1] = 6 * (X[:, 0] + X[:, 1] > 0) * (-X[:, 0] + X[:, 1] < 0) 
        trt_panel[:, 2] = 5 * (X[:, 0] + X[:, 1] < 0) * (-X[:, 0] + X[:, 1] > 0)
        trt_panel[:, 3] = 3 * (X[:, 0] + X[:, 1] > 0) * (-X[:, 0] + X[:, 1] > 0)

    elif case == 2:

        A = np.random.choice([-1, 1], (sample_size, 2))
        A_cate = _categorical_treatment(A)

        trt_panel = np.zeros((sample_size, 4))
        trt_panel[:, 0] = X[:, 0] + 2 * X[:, 3] - 2 * X[:, 4]
        trt_panel[:, 1] = - X[:, 1] + 2 * X[:, 2] - 2 * X[:, 4]
        trt_panel[:, 2] = X[:, 2] - 3 * X[:, 3] + 1/2 * X[:, 4]
        trt_panel[:, 3] = - X[:, 0] - 2 * X[:, 3] + 2 * X[:, 6]
        
    elif case == 3:

        A = np.random.choice([-1, 1], (sample_size, 3))
        A_cate = _categorical_treatment(A)

        trt_panel = np.zeros((sample_size, 8))
        trt_panel[:, 0] = 1/2 * (X[:, 0] ** 2 - X[:, 1] * X[:, 2])
        trt_panel[:, 1] = 2 * (X[:, 0] + np.exp(X[:, 1]))
        trt_panel[:, 2] = X[:, 2] + (X[:, 3] + X[:, 4]) ** 2
        trt_panel[:, 3] = np.log((X[:, 4] + 1) ** 2)
        trt_panel[:, 4] = np.exp(X[:, 5] + X[:, 6])
        trt_panel[:, 5] = X[:, 7] + X[:, 8] + X[:, 9]
        trt_panel[:, 6] = X[:, 0] + (X[:, 1] - X[:, 2]) ** 3
        trt_panel[:, 7] = (X[:, 0] - X[:, 4] + X[:, 5]) ** 2

    Y = m + trt_panel[np.arange(sample_size), A_cate] + np.random.normal(0, 0.3, sample_size)

    if outcome_shift is not None:

        Y = Y - np.min(Y) + outcome_shift

    A_uni = np.unique(A, axis=0)
    _optA = np.argmax(trt_panel, axis=1)    
    optA = A_uni[_optA, :]

    return X, A, Y, optA



        