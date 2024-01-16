import numpy as np

def combinize_treatment(A, K):
    
    """
    Create combinatorial representation of categorical treatment
    """

    pairs = np.array(np.meshgrid(*[[-1, 1]] * K)).T.reshape(-1, K)

    A_comb = pairs[A]

    return A_comb

def categorize_treatment(A, K):

    """
    Create categorical representation of combination treatment
    """

    pairs = np.array(np.meshgrid(*[[-1, 1]] * K)).T.reshape(-1, K)

    A_cate = np.zeros(A.shape[0], dtype=int)

    for j in range(2 ** K):

        A_cate[np.all(A == pairs[j, :], axis=1)] = j

    return A_cate


def getdata(sample_size, case=1, outcome_shift=None, 
            seed=None, randomized=True):

    if seed is not None:

        np.random.seed(seed)

    X = np.random.uniform(-1, 1, (sample_size, 10))

    m = 1 + X[:, 0] + 2 * X[:, 1]

    beta = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]

    if case == 1:

        if randomized:
        
            A = np.random.choice([-1, 1], (sample_size, 2))
            A_cate = categorize_treatment(A, 2)

        else:

            prob = np.zeros((sample_size, 4))

            for j in range(4):

                prob[:, j] = np.exp(X.dot(beta) * j)

            prob = prob / np.sum(prob, axis=1).reshape(-1, 1)

            A_cate = np.zeros((sample_size, ), dtype=int)

            for i in range(sample_size):

                A_cate[i] = np.random.choice([0, 1, 2, 3], size = 1, p = prob[i, :])[0]

            A = combinize_treatment(A_cate, 2)

        trt_panel = np.zeros((sample_size, 4))
        trt_panel[:, 0] = 0 * (X[:, 0] + X[:, 1] < 0) * (-X[:, 0] + X[:, 1] < 0)
        trt_panel[:, 1] = 6 * (X[:, 0] + X[:, 1] > 0) * (-X[:, 0] + X[:, 1] < 0) 
        trt_panel[:, 2] = 5 * (X[:, 0] + X[:, 1] < 0) * (-X[:, 0] + X[:, 1] > 0)
        trt_panel[:, 3] = 3 * (X[:, 0] + X[:, 1] > 0) * (-X[:, 0] + X[:, 1] > 0)

    elif case == 2:

        if randomized:
            
            A = np.random.choice([-1, 1], (sample_size, 2))
            A_cate = categorize_treatment(A, 2)

        else:

            prob = np.zeros((sample_size, 4))

            for j in range(4):

                prob[:, j] = np.exp(X.dot(beta) * j)

            prob = prob / np.sum(prob, axis=1).reshape(-1, 1)

            A_cate = np.zeros((sample_size, ), dtype=int)

            for i in range(sample_size):

                A_cate[i] = np.random.choice([0, 1, 2, 3], size = 1, p = prob[i, :])[0]

            A = combinize_treatment(A_cate, 2)


        trt_panel = np.zeros((sample_size, 4))
        trt_panel[:, 0] = (X[:, 0] + X[:, 1]) ** 2
        trt_panel[:, 1] = X[:, 1] ** 2 + X[:, 2] * X[:, 3]
        trt_panel[:, 2] = -X[:, 2] * X[:, 3]
        trt_panel[:, 3] = X[:, 1] ** 2 + 3 * X[:, 4] * X[:, 5]
        
    elif case == 3:

        if randomized:
            
            A = np.random.choice([-1, 1], (sample_size, 3))
            A_cate = categorize_treatment(A, 3)

        else:

            prob = np.zeros((sample_size, 8))

            for j in range(8):

                prob[:, j] = np.exp(X.dot(beta) * j)

            prob = prob / np.sum(prob, axis=1).reshape(-1, 1)

            A_cate = np.zeros((sample_size, ), dtype=int)

            for i in range(sample_size):

                A_cate[i] = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size = 1, p = prob[i, :])[0]

            A = combinize_treatment(A_cate, 3)


        trt_panel = np.zeros((sample_size, 8))
        trt_panel[:, 0] = 0
        trt_panel[:, 1] = 2 * (X[:, 0] + np.exp(X[:, 1]))
        trt_panel[:, 2] = X[:, 2] + (X[:, 3] + X[:, 4]) ** 2
        trt_panel[:, 3] = 2 * (X[:, 0] + np.exp(X[:, 1])) + X[:, 2] + (X[:, 3] + X[:, 4]) ** 2 + np.log((X[:, 4] + 1) ** 2)
        trt_panel[:, 4] = np.exp(X[:, 5] + X[:, 6])
        trt_panel[:, 5] = np.exp(X[:, 5] + X[:, 6]) + 2 * (X[:, 0] + np.exp(X[:, 1])) + X[:, 7] + X[:, 8] + X[:, 9]
        trt_panel[:, 6] = np.exp(X[:, 5] + X[:, 6]) + X[:, 2] + (X[:, 3] + X[:, 4]) ** 2
        trt_panel[:, 7] = np.exp(X[:, 5] + X[:, 6]) + X[:, 2] + (X[:, 3] + X[:, 4]) ** 2 + 2 * (X[:, 0] + np.exp(X[:, 1])) + (X[:, 0] - X[:, 4] + X[:, 5]) ** 2

    Y = m + trt_panel[np.arange(sample_size), A_cate] + np.random.normal(0, 0.3, sample_size)

    if outcome_shift is not None:

        Y = Y - np.min(Y) + outcome_shift

    _optA = np.argmax(trt_panel, axis=1)

    if case == 3:    

        optA = combinize_treatment(_optA, 3)

    else:

        optA = combinize_treatment(_optA, 2)

    return X, A, Y, optA



        