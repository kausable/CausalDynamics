from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class NGC_LSTM:
    """
    Nural GC (LSTM-variant) baseline.

    Reference:
        [1] https://github.com/iancovert/Neural-GC/blob/master/clstm_lorenz_demo.ipynb
    """

    def __init__(self, tau_max: int = 1):
        """Initialize regressor"""
        super(NGC_LSTM, self).__init__()
        self.tau_max = tau_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, X, verbosity: int = 0, hidden_dim: int = 16):
        """Estimate lagged adjacency graph"""

        X = torch.tensor(X[None,]).to(self.device)
        self.estimator = cLSTM(X.shape[-1], hidden=hidden_dim).to(self.device)
        self.train_loss = train_model_ista(
            self.estimator,
            X,
            context=1,
            tau=self.tau_max,
            lam=10.0,
            lam_ridge=1e-2,
            lr=1e-3,
            max_iter=100,
            check_every=1,
            verbose=0,
        )
        self.train_loss = torch.tensor(self.train_loss).detach().cpu().numpy()
        self.adj_matrix = self.estimator.GC().T.data.detach().cpu().numpy()


class LSTM(nn.Module):
    def __init__(self, num_series, hidden):
        """
        LSTM model with output layer to generate predictions.

        Args:
          num_series: number of input time series.
          hidden: number of hidden units.
        """
        super(LSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.lstm = nn.LSTM(num_series, hidden, batch_first=True)
        self.lstm.flatten_parameters()
        self.linear = nn.Conv1d(hidden, 1, 1)

    def init_hidden(self, batch):
        """Initialize hidden states for LSTM cell."""
        device = self.lstm.weight_ih_l0.device
        return (
            torch.zeros(1, batch, self.hidden, device=device),
            torch.zeros(1, batch, self.hidden, device=device),
        )

    def forward(self, X, hidden=None):
        # Set up hidden state.
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])

        # Apply LSTM.
        X, hidden = self.lstm(X, hidden)

        # Calculate predictions using output layer.
        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1), hidden


class cLSTM(nn.Module):
    def __init__(self, num_series, hidden):
        """
        cLSTM model with one LSTM per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          hidden: number of units in LSTM cell.
        """
        super(cLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up networks.
        self.networks = nn.ModuleList(
            [LSTM(num_series, hidden) for _ in range(num_series)]
        )

    def forward(self, X, hidden=None):
        """
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
          hidden: hidden states for LSTM cell.
        """
        if hidden is None:
            hidden = [None for _ in range(self.p)]
        pred = [self.networks[i](X, hidden[i]) for i in range(self.p)]
        pred, hidden = zip(*pred)
        pred = torch.cat(pred, dim=2)
        return pred, hidden

    def GC(self, threshold=True):
        """
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        """
        GC = [torch.norm(net.lstm.weight_ih_l0, dim=0) for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


def prox_update(network, lam, lr):
    """Perform in place proximal update on first layer weight matrix."""
    W = network.lstm.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = (W / torch.clamp(norm, min=(lam * lr))) * torch.clamp(
        norm - (lr * lam), min=0.0
    )
    network.lstm.flatten_parameters()


def regularize(network, lam):
    """Calculate regularization term for first layer weight matrix."""
    W = network.lstm.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    """Apply ridge penalty at linear layer and hidden-hidden weights."""
    return lam * (
        torch.sum(network.linear.weight**2) + torch.sum(network.lstm.weight_hh_l0**2)
    )


def restore_parameters(model, best_model):
    """Move parameter values from best_model to model."""
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context, tau=1):
    """
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    """
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(
        len(data) - context,
        context,
        data.shape[1],
        dtype=torch.float32,
        device=data.device,
    )
    target = torch.zeros(
        len(data) - context,
        context,
        data.shape[1],
        dtype=torch.float32,
        device=data.device,
    )
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start + tau : end + tau]
    return input.detach(), target.detach()


def train_model_ista(
    clstm,
    X,
    context,
    tau,
    lr,
    max_iter,
    lam=0,
    lam_ridge=0,
    lookback=5,
    check_every=50,
    verbose=1,
):
    """Train model with Adam."""
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction="mean")
    train_loss_list = []

    # Set up data.
    X, Y = zip(*[arrange_input(x, context=context, tau=tau) for x in X])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    pred = [clstm.networks[i](X)[0] for i in range(p)]
    loss = sum([loss_fn(pred[i][:, :, 0], Y[:, :, i]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in clstm.networks])
    smooth = loss + ridge

    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in clstm.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in clstm.networks:
                prox_update(net, lam, lr)

        clstm.zero_grad()

        # Calculate loss for next iteration.
        pred = [clstm.networks[i](X)[0] for i in range(p)]
        loss = sum([loss_fn(pred[i][:, :, 0], Y[:, :, i]) for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in clstm.networks])
        smooth = loss + ridge

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            nonsmooth = sum([regularize(net, lam) for net in clstm.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(("-" * 10 + "Iter = %d" + "-" * 10) % (it + 1))
                print("Loss = %f" % mean_loss)
                print(
                    "Variable usage = %.2f%%" % (100 * torch.mean(clstm.GC().float()))
                )

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(clstm)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print("Stopping early")
                break

    # Restore best model.
    restore_parameters(clstm, best_model)

    return train_loss_list
