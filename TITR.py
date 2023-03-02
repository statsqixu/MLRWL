import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.linear_model import LinearRegression
from util import getdata

class TITR():

    def __init__(self, kernel="linear", gamma=None, sigma=None, degree=None):

        self.kernel = kernel
        self.gamma = gamma
        self.sigma = sigma
        self.degree = degree

    def fit(self, X, A, Y, max_iter=100):

        n, p = X.shape

        k = A.shape[1]

        if self.kernel == "linear":
    
            self.K = X.dot(X.T)

            self.X_train = X

            # initialize sub_grad

            sub_grad = np.zeros((p, k))
            beta_cur = np.zeros((p, k))


        elif self.kernel == "rbf":
    
            self.K = rbf_kernel(X, gamma=1/(self.sigma ** 2))

            self.X_train = X

            # initialize sub_grad

            sub_grad = np.zeros((n, k))
            beta_cur = np.zeros((n, k))


        elif self.kernel == "poly":

            self.K = polynomial_kernel(X, degree=self.degree)

            self.X_train = X

            # initialize sub_grad

            sub_grad = np.zeros((n, k))
            beta_cur = np.zeros((n, k))

        # estimate treatment-free effects

        lm = LinearRegression(fit_intercept=True)
        tf = lm.fit(X, Y).predict(X)

        R = Y - tf

        W = self.gamma * R

        for _ in range(max_iter):

            beta_prev = beta_cur

            beta_cur = self.qp_solver(A, W, sub_grad)

            err = np.linalg.norm(beta_cur - beta_prev, ord="fro")

            if err < 1e-4:

                break
            
            if self.kernel == "linear":
    
                predictor1 = 1 - (X.dot(beta_cur)) * A
                predictor2 = - (X.dot(beta_cur)) * A

            else:

                predictor1 = 1 - (self.K.dot(beta_cur)) * A
                predictor2 = - (self.K.dot(beta_cur)) * A

            indicator1 = np.argmax(np.c_[predictor1, np.zeros(n)], axis=1)
            indicator1[W > 0] = k + 1

            indicator2 = np.argmax(np.c_[predictor2, np.zeros(n)], axis=1)
            indicator2[W < 0] = k + 1

            sub_grad = self.sub_gradient(A, W, indicator1, indicator2)

        self.beta = beta_cur

        return beta_cur

    def predict(self, X_test):

        if self.kernel == "linear":

            D = np.sign(X_test.dot(self.beta))

        elif self.kernel == "rbf":

            K_test = rbf_kernel(X_test, self.X_train, gamma=1/(self.sigma ** 2))

            D = np.sign(K_test.dot(self.beta))

        elif self.kernel == "poly":

            K_test = polynomial_kernel(X_test, self.X_train, degree=self.degree)

            D = np.sign(K_test.dot(self.beta))

        return D

    def evaluate(self, Y_test, A_test, D_test, optA_test=None):

        output = []

        if optA_test is not None:

            acc = np.mean((D_test == optA_test).all(axis=1))

            output.append(acc)

        val = np.sum(Y_test * (D_test == A_test).all(axis=1)) / np.sum((D_test == A_test).all(axis=1))

        output.append(val)

        return output

    def qp_solver(self, A, W, sub_grad):

        n = self.K.shape[0]
        p = self.X_train.shape[1]

        k = A.shape[1]

        if self.kernel == "linear":

            Q = block_diag(*[(self.X_train * A[:, kk][:, np.newaxis]).dot((self.X_train * A[:, kk][:, np.newaxis]).transpose()) for kk in range(k)])

            P = - np.concatenate([(self.X_train * A[:, kk][:, np.newaxis]).dot(sub_grad[:, kk][:, np.newaxis]) for kk in range(k)], axis=0)

        else:

            Q = block_diag(*[np.diag(A[:, kk]).dot(self.K).dot(np.diag(A[:, kk])) for kk in range(k)])

            P = - np.concatenate([np.diag(A[:, kk]).dot(sub_grad[:, kk]) for kk in range(k)], axis=0)

        C1 = np.concatenate([np.eye(n) for _ in range(k)], axis=1)

        D1 = np.tile(1 * (W > 0), k)

        m = gp.Model("qp")

        lambd = m.addMVar((n * k), lb = 0.0, vtype = GRB.CONTINUOUS, name = "lambd")
        m.setParam("OutputFlag", 0)
        m.setObjective(1/2 * lambd.T @ Q @ lambd + P.T @ lambd - D1.T @ lambd, GRB.MINIMIZE)
        m.addConstr(C1 @ lambd <= np.abs(W))

        m.optimize()

        lambd = lambd.X

        if self.kernel == "linear":

            beta = np.zeros((p, k))

        else:

            beta = np.zeros((n, k))

        for kk in range(k):

            if self.kernel == "linear":

                beta[:, kk] = - sub_grad[:, kk] + (self.X_train * A[:, kk][:, np.newaxis]).T.dot(lambd[kk * n: (kk + 1) * n])

            else: 
                
                beta[:, kk] = np.linalg.inv(self.K).dot(-sub_grad[:, kk] + np.diag(A[:, kk]).dot(self.K).dot(lambd[kk * n: (kk + 1) * n]))

        return beta

    def sub_gradient(self, A, W, indicator1, indicator2):

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

                        if indicator1[i] == kk:
    
                            sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.X_train[i, :]

                        elif indicator2[i] == kk:

                            sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.X_train[i, :]

                    else: 
                        
                        if indicator1[i] == kk:
        
                            sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.K[:, i]
                        
                        elif indicator2[i] == kk:

                            sub_grad[:, kk] += np.abs(W[i]) * A[i, kk] * self.K[:, i]

        return sub_grad
