import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.linear_model import LinearRegression, LogisticRegression
from util import *
import cvxpy as cp

class TITR():

    def __init__(self, kernel="linear", gamma=None, sigma=None, degree=None):

        self.kernel = kernel
        self.gamma = gamma
        self.sigma = sigma
        self.degree = degree

    def fit(self, X, A, Y, max_iter=100, verbose=0):

        n, p = X.shape

        k = A.shape[1]

        np.random.seed(seed=None)

        if self.kernel == "linear":
    
            self.K = X.dot(X.T)

            self.X_train = X

            # initialize sub_grad

            sub_grad_cur = np.zeros((p, k))
            sub_grad0_cur = np.zeros((1, k))
            beta_cur = np.zeros((p, k))
            beta0_cur = np.zeros((1, k))

        elif self.kernel == "rbf":
    
            self.K = rbf_kernel(X, gamma=1/(self.sigma ** 2))

            self.X_train = X

            # initialize sub_grad

            sub_grad_cur = np.zeros((n, k))
            sub_grad0_cur = np.zeros((1, k))
            beta_cur = np.zeros((n, k))
            beta0_cur = np.zeros((1, k))

        elif self.kernel == "poly":

            self.K = polynomial_kernel(X, degree=self.degree)

            self.X_train = X

            # initialize sub_grad

            sub_grad_cur = np.zeros((n, k))
            sub_grad0_cur = np.zeros((1, k))
            beta_cur = np.zeros((n, k))
            beta0_cur = np.zeros((1, k))

        # estimate treatment-free effects

        lm = LinearRegression(fit_intercept=True)
        tf = lm.fit(X, Y).predict(X)

        mlm = LogisticRegression(multi_class="multinomial")
        ps = mlm.fit(X, categorize_treatment(A, A.shape[1])).predict_proba(X)
        ps = ps[np.arange(A.shape[0]), categorize_treatment(A, A.shape[1])]

        R = Y - tf

        W = self.gamma * R / ps

        for _ in range(max_iter):

            beta_prev = beta_cur
            beta0_prev = beta0_cur
            sub_grad_prev = sub_grad_cur
            sub_grad0_prev = sub_grad0_cur

            lambd = self.qp_solver(A, W, sub_grad_cur, sub_grad0_cur)

            beta_cur = self.compute_beta(lambd, A, sub_grad_cur)

            beta0_cur = self.compute_beta0(lambd, A, W, beta_cur)

            sub_grad_cur = self.compute_sub_grad(A, W, beta_cur, beta0_cur)

            sub_grad0_cur = self.compute_sub_grad0(A, W, beta_cur, beta0_cur)

            err = max(np.linalg.norm(beta_cur - beta_prev, ord="fro") + np.linalg.norm(beta0_cur - beta0_prev, ord="fro"), 
                        np.linalg.norm(sub_grad_cur - sub_grad_prev, ord="fro") + np.linalg.norm(sub_grad0_cur - sub_grad0_prev, ord="fro"))

            if verbose == 1:

                print("iter: {}, error: {}".format(_, err))

            if err < 1e-6:

                break

        self.beta = beta_cur
        self.beta0 = beta0_cur

        return beta_cur, beta0_cur

    def predict(self, X_test):

        n, p = X_test.shape

        if self.kernel == "linear":

            D = np.sign(X_test.dot(self.beta) + self.beta0)

        elif self.kernel == "rbf":

            K_test = rbf_kernel(X_test, self.X_train, gamma=1/(self.sigma ** 2))

            D = np.sign(K_test.dot(self.beta) + self.beta0)

        elif self.kernel == "poly":

            K_test = polynomial_kernel(X_test, self.X_train, degree=self.degree)

            D = np.sign(K_test.dot(self.beta) + self.beta0)

        return D

    def evaluate(self, Y_test, A_test, D_test, optA_test=None, randomized=True, X_test=None):

        output = []

        if optA_test is not None:

            acc = np.mean((D_test == optA_test).all(axis=1))

            output.append(acc)

        val = np.sum(Y_test * (D_test == A_test).all(axis=1)) / np.sum((D_test == A_test).all(axis=1))

        output.append(val)

        return output

    def qp_solver(self, A, W, sub_grad, sub_grad0):

        n = self.K.shape[0]

        k = A.shape[1]

        if self.kernel == "linear":

            Q = block_diag(*[(self.X_train * A[:, kk][:, np.newaxis]).dot((self.X_train * A[:, kk][:, np.newaxis]).transpose()) for kk in range(k)])

            P = - np.concatenate([(self.X_train * A[:, kk][:, np.newaxis]).dot(sub_grad[:, kk][:, np.newaxis]) for kk in range(k)], axis=0).T

        else:

            Q = block_diag(*[np.diag(A[:, kk]).dot(self.K).dot(np.diag(A[:, kk])) for kk in range(k)])

            P = - np.concatenate([np.diag(A[:, kk]).dot(sub_grad[:, kk]) for kk in range(k)], axis=0).T

        O = np.tile(1 * (W > 0), k).T
        
        C1 = np.concatenate([np.eye(n) for _ in range(k)], axis=1)
        C2 = block_diag(*[A[:, kk] for kk in range(k)])

        m = gp.Model("qp")

        lambd = m.addMVar((n * k), lb = 0.0, vtype = GRB.CONTINUOUS, name = "lambd")
        m.setParam("OutputFlag", 0)
        m.setObjective(1/2 * lambd.T @ (Q) @ lambd + self.gamma * P @ lambd - O @ lambd, GRB.MINIMIZE)
        m.addConstr(C1 @ lambd <= np.abs(W))
        m.addConstr(C2 @ lambd == self.gamma * sub_grad0)

        m.optimize()

        lambd = lambd.X
        
        return lambd
    
    def compute_beta(self, lambd, A, sub_grad):

        n = self.K.shape[0]
        p = self.X_train.shape[1]
        k = A.shape[1]

        if self.kernel == "linear":

            beta = np.zeros((p, k))

            for kk in range(k):

                beta[:, kk] = np.sum(np.diag(A[:, kk] * lambd[kk * n: (kk + 1) * n]).dot(self.X_train), axis=0) - sub_grad[:, kk]

        else:

            beta = np.zeros((n, k))

            for kk in range(k):

                beta[:, kk] = A[:, kk] * lambd[kk * n: (kk + 1) * n] - np.linalg.inv(self.K).dot(sub_grad[:, kk])

        return beta
    
    def compute_beta0(self, lambd, A, W, beta):

        n = self.K.shape[0]
        p = self.X_train.shape[1]
        k = A.shape[1]

        lambd_sum = np.sum(lambd.reshape((k, n)), axis=0)

        index = np.where(lambd_sum < W)[0]

        beta0 = np.zeros((1, k))

        for kk in range(k):

            lambd_sub = lambd[kk * n: (kk + 1) * n]
            index_sub = np.where(lambd_sub > 0)[0]

            # take the intersection of index and index_sub

            index_int = np.intersect1d(index, index_sub)

            if self.kernel == "linear":

                beta0[0, kk] = np.mean(A[index_int, kk] * (W[index_int] > 0) - self.X_train[index_int, :].dot(beta[:, kk]))

            else:

                beta0[0, kk] = np.mean(A[index_int, kk] * (W[index_int] > 0) - self.K[index_int, :].dot(beta[:, kk]))

        return beta0

    def compute_sub_grad(self, A, W, beta_cur, beta0_cur):

        n = self.K.shape[0]
        k = A.shape[1]

        if self.kernel == "linear":
            
            p = self.X_train.shape[1]
            sub_grad = np.zeros((p, k))

        else:
            sub_grad = np.zeros((n, k))


        for i in range(n):

            for kk in range(k):

                if self.kernel == "linear":

                    predictor1 = np.array([1 - A[:, kk] * (self.X_train.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor1 = np.c_[predictor1, np.zeros((n, ))]
                    indicator1 = np.argmax(predictor1, axis=1)

                    predictor2 = np.array([- A[:, kk] * (self.X_train.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor2 = np.c_[predictor2, np.zeros((n, ))]
                    indicator2 = np.argmax(predictor2, axis=1)

                    if indicator1[i] == kk:

                        sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.X_train[i, :] * (W[i] < 0)

                    elif indicator2[i] == kk:

                        sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.X_train[i, :] * (W[i] <= 0)

                else: 
                    
                    predictor1 = np.array([1 - A[:, kk] * (self.K.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor1 = np.c_[predictor1, np.zeros((n, ))]
                    indicator1 = np.argmax(predictor1, axis=1)

                    predictor2 = np.array([- A[:, kk] * (self.K.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor2 = np.c_[predictor2, np.zeros((n, ))]
                    indicator2 = np.argmax(predictor2, axis=1)

                    if indicator1[i] == kk:
    
                        sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.K[:, i] * (W[i] < 0)
                    
                    elif indicator2[i] == kk:

                        sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.K[:, i] * (W[i] >= 0)

        return sub_grad
    
    def compute_sub_grad0(self, A, W, beta_cur, beta0_cur):

        n = self.K.shape[0]
        k = A.shape[1]

        if self.kernel == "linear":
            
            p = self.X_train.shape[1]
            sub_grad0 = np.zeros((1, k))

        else:
            sub_grad0 = np.zeros((1, k))


        for i in range(n):

            for kk in range(k):

                if self.kernel == "linear":

                    predictor1 = np.array([1 - A[:, kk] * (self.X_train.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor1 = np.c_[predictor1, np.zeros((n, ))]
                    indicator1 = np.argmax(predictor1, axis=1)

                    predictor2 = np.array([- A[:, kk] * (self.X_train.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor2 = np.c_[predictor2, np.zeros((n, ))]
                    indicator2 = np.argmax(predictor2, axis=1)

                    if indicator1[i] == kk:

                        sub_grad0[0, kk] += np.abs(W[i]) * A[i, kk] * (W[i] < 0)

                    elif indicator2[i] == kk:

                        sub_grad0[0, kk] += np.abs(W[i]) * A[i, kk] * (W[i] >= 0)

                else: 
                    
                    predictor1 = np.array([1 - A[:, kk] * (self.K.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor1 = np.c_[predictor1, np.zeros((n, ))]
                    indicator1 = np.argmax(predictor1, axis=1)

                    predictor2 = np.array([- A[:, kk] * (self.K.dot(beta_cur[:, kk]) + beta0_cur[0, kk]) for kk in range(k)]).T
                    predictor2 = np.c_[predictor2, np.zeros((n, ))]
                    indicator2 = np.argmax(predictor2, axis=1)

                    if indicator1[i] == kk:
    
                        sub_grad0[0, kk] += np.abs(W[i]) * A[i, kk] * (W[i] < 0)
                    
                    elif indicator2[i] == kk:

                        sub_grad0[0, kk] += np.abs(W[i]) * A[i, kk] * (W[i] >= 0)

        return sub_grad0
