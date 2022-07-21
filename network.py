from pickletools import optimize
from numpy.lib.npyio import NpzFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import hamming_loss as hml
import numpy as np

class MCDNet(nn.Module):

    def __init__(self, input_size, output_size=3, layer=2, act="linear",
                 width=20, loss="ghl"):

        super(MCDNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.layer = layer
        self.act = act
        self.width = width
        self.loss = loss

        self.input = nn.Linear(input_size, width)

        self.hidden = nn.ModuleList()
        for i in range(layer):
            self.hidden.append(nn.Linear(width, width))
            self.hidden.append(nn.BatchNorm1d(width))

        self.output = nn.Linear(width, output_size)

    def forward(self, X):

        cov = self.input(X)

        if self.act == "relu":

            cov = F.relu(cov)

        elif self.act == "linear":

            cov = cov

        for index, layer in enumerate(self.hidden):

            if index % 2 == 0:

                cov = layer(cov)
                break

            elif index % 2 == 1:

                cov = layer(cov)

                if self.act == "relu":

                    cov = F.relu(cov)

                elif self.act == "linear":

                    cov = cov

        dec_func = self.output(cov)

        return dec_func

    def generalized_hinge_loss(self, dec, A, Y):

        Z = torch.mul(dec, A)

        phi = torch.min(Z - 1)

        phi = torch.minimum(phi, torch.tensor(0, dtype=torch.int16))
        loss = - (Y * phi).mean()

        return loss


    def hamming_loss(self, dec, A, Y):

        y_true = A.detach().numpy()

        y_pred = dec.detach().numpy()

        y_true = np.argmax(y_true, axis=1)

        y_pred = np.argmax(y_pred, axis=1)

        phi = hml(y_true, y_pred)

        loss = -(Y * phi).mean()

        return loss

    def training_step(self, batch):

        Y, X, A = batch

        dec_func = self(X)

        # 

        if self.loss == "hgl":

            loss = self.generalized_hinge_loss(dec_func, A, Y)

        elif self.loss == "hml":

            loss = self.hamming_loss(dec_func, A, Y)

        return loss

    def epoch_end(self, epoch, result):

        print("Epoch: {} - Training loss: {:.4f}".format(epoch, result))


class Trainer():

    def fit(self, loss, epochs, learning_rate, model, train_loader, print_history,
            opt_func, weight_decay, device):

        history = []

        optimizer = opt_func(model.parameters(), learning_rate,
                             weight_decay=weight_decay)
        optimizer.zero_grad()
        scheduler = ExponentialLR(optimizer, gamma=0.999)

        for epoch in range(epochs):

            for batch in train_loader:

                batch = [item.to(device) for item in batch]
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            result = self._evaluate(model, train_loader, device)

            if print_history:
                model.epoch_end(epoch, result)

            history.append(result)

        return history

    def _evaluate(self, model, train_loader, device):

        outputs = []

        for batch in train_loader:

            batch = [item.to(device) for item in batch]
            outputs.append(model.training_step(batch))

        return torch.stack(outputs).mean()
