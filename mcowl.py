
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from network2 import MCDNet, Trainer
from container import ITRDataset
from util import plot_train_history

def _return_device(device):

    if device == "cpu":

        return "cpu"

    elif device == "gpu":

        return "cuda:0"

    elif device == "default":

        if torch.cuda.is_available():

            return "cuda:0"

        else:

            return "cpu"

class MCOWL():

    def __init__(self, act="relu", layer=2, width=20, loss="ghl"):

        self.act = act
        self.layer = layer
        self.width = width
        self.loss = loss

    def fit(self, Y, X, A, Yt, Xt, At, optAt, epochs=100, learning_rate=1e-3, verbose=0, opt_func=Adam, weight_decay=0.01, 
                    batch_size=32, dp=0.4, device="default"):

        _device = _return_device(device)

        if verbose > 0:

            print("------ The program is running on {0} ------".format(_device))

        self.device = device

        input_dim = X.shape[1]
        n_samples = X.shape[0]
        output_dim = A.shape[1]

        self.model = MCDNet(input_dim, output_dim, self.layer, self.act, self.width, self.loss, dp)

        self.model = self.model.to(self.device)

        Y_tsr = torch.from_numpy(Y).float()
        X_tsr = torch.from_numpy(X).float()
        A_tsr = torch.from_numpy(A).float()
        Yt_tsr = torch.from_numpy(Yt).float()
        Xt_tsr = torch.from_numpy(Xt).float()
        At_tsr = torch.from_numpy(At).float()
        optAt_tsr = torch.from_numpy(optAt).float()

        dataset = ITRDataset(Y_tsr, X_tsr, A_tsr)
        dataset_t=ITRDataset(Yt_tsr, Xt_tsr, At_tsr)

        loader = DataLoader(dataset, batch_size=batch_size)
        loader_t = DataLoader(dataset_t, batch_size=batch_size)

        if verbose == 0:

            print_history = False
            plot_history = False

        elif verbose == 1:

            print_history = True
            plot_history = False

        elif verbose == 2:

            print_history = True
            plot_history = False

        elif verbose == 3:

            print_history = True
            plot_history = True

        trainer = Trainer()
        
        if self.loss=='ghl':
          np.random.seed()
          print('opt loss:',self.model.generalized_hinge_loss(optAt_tsr,At_tsr,Yt_tsr))
          print('random loss:',self.model.generalized_hinge_loss(torch.from_numpy(np.random.choice([-1,1],(n_samples,output_dim))).float(),At_tsr,Yt_tsr))
        elif self.loss=='hml':
          np.random.seed()
          print('opt loss:',self.model.hamming_loss(optAt_tsr,At_tsr,Yt_tsr))
          print('random loss:',self.model.hamming_loss(torch.from_numpy(np.random.choice([-1,1],(n_samples,output_dim))).float(),At_tsr,Yt_tsr))
        
        history = []
        history += trainer.fit(epochs=epochs, learning_rate=learning_rate, model=self.model, train_loader=loader,test_loader=loader_t,
                                print_history=print_history, opt_func=opt_func, weight_decay=weight_decay, device=self.device)
        
        if plot_history:

            plot_train_history(history)

        return history

    def predict(self, X):

        X_tsr = torch.from_numpy(X).float().to(self.device)
        self.model.eval()
        dec = self.model(X_tsr)
        D = torch.sign(dec)

        return D.detach().numpy()

    def evaluate(self, Y, A, D, optA=None, accuracy=True, value=True):

        Y = torch.from_numpy(Y).float()
        A = torch.from_numpy(A).float()
        D = torch.from_numpy(D).float()

        n_samples = len(Y)

        prop = np.ones((n_samples, ))

        output = []

        if accuracy:

            if optA is None:

                raise Exception("Optimal assignment is unknown.")

            else:

                optA = torch.from_numpy(optA).float()
                acc = torch.mean(torch.all(D == optA, dim=1) * 1.0)
                output.append(acc)

        if value:

            nom = torch.sum(torch.all(D == A, dim=1) * Y / prop) / n_samples
            den = torch.sum(torch.all(D == A, dim=1) * 1.0 / prop) / n_samples

            val = nom / den
            output.append(val.numpy())

        return output




    
