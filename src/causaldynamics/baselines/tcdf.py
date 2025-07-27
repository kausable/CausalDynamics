import warnings
import random
import copy
import heapq

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

warnings.filterwarnings("ignore")


class TCDF:
    """
    TCDF baseline wrapper that accepts in-memory numpy arrays.

    Reference:
        Nauta, M. et al. “Temporal Causal Discovery Framework (TCDF)”
        https://github.com/M-Nauta/TCDF
    """
    def __init__(
        self,
        cuda: bool = False,
        epochs: int = 100,
        kernel_size: int = 4,
        hidden_layers: int = 1,
        learning_rate: float = 0.01,
        optimizer: str = "Adam",
        seed: int = 1111,
        dilation_coefficient: int = 4,
        significance: float = 0.8,
        log_interval: int = 500,
    ):
        self.cuda = cuda
        self.epochs = epochs
        self.kernel_size = kernel_size
        self.layers = hidden_layers + 1  # TCDF uses levels = hidden_layers + 1
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.seed = seed
        self.dilation_coefficient = dilation_coefficient
        self.significance = significance
        self.log_interval = log_interval

    def _prepare_data(self, df: pd.DataFrame, target: str):
        df_y = df[[target]].copy()
        df_x = df.copy()
        df_yshift = df_y.shift(1).fillna(0.0)
        df_x[target] = df_yshift[target]
        data_x = df_x.values.astype("float32").T  # (channels, seq_len)
        data_y = df_y.values.astype("float32").T  # (1, seq_len)
        x = Variable(torch.from_numpy(data_x))
        y = Variable(torch.from_numpy(data_y))
        return x, y

    def _train_epoch(self, epoch, X_train, Y_train, model, optimizer):
        model.train()
        x, y = X_train[0:1], Y_train[0:1]
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        return model.fs_attention.data.clone(), loss.item()

    def _findcauses(self, df: pd.DataFrame, target: str):
        torch.manual_seed(self.seed)
        X_train, Y_train = self._prepare_data(df, target)
        X_train = X_train.unsqueeze(0)
        Y_train = Y_train.unsqueeze(2)
        input_channels = X_train.size(1)
        targetidx = int(target)

        model = ADDSTCN(targetidx, input_channels,
                        self.layers, self.kernel_size,
                        self.cuda, self.dilation_coefficient)
        if self.cuda:
            model.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
        optimizer = getattr(optim, self.optimizer)(model.parameters(), lr=self.learning_rate)

        att, firstloss = self._train_epoch(1, X_train, Y_train, model, optimizer)
        for ep in range(2, self.epochs + 1):
            att, loss = self._train_epoch(ep, X_train, Y_train, model, optimizer)
        realloss = loss

        scores = att.view(-1).cpu().numpy()
        idx_sorted = np.argsort(-scores)
        s_sorted = np.sort(scores)[::-1]

        if len(s_sorted) <= 5:
            potentials = [i for i in idx_sorted if scores[i] > 1.0]
        else:
            gaps = []
            for i in range(len(s_sorted) - 1):
                if s_sorted[i] < 1.0:
                    break
                gaps.append(s_sorted[i] - s_sorted[i+1])
            potentials = idx_sorted[: gaps.index(max(gaps)) + 1].tolist() if gaps else []

        validated = potentials.copy()
        for v in potentials:
            tmp = X_train.clone().cpu().numpy()
            random.shuffle(tmp[0, v, :])
            shuffled = torch.from_numpy(tmp)
            if self.cuda:
                shuffled = shuffled.cuda()
            model.eval()
            out = model(shuffled)
            testloss = F.mse_loss(out, Y_train).item()
            if (firstloss - testloss) <= (firstloss - realloss) * self.significance:
                validated.remove(v)

        return validated

    def run(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate directed adjacency from array X of shape (T, n_vars).

        Returns:
            adj_matrix: boolean array (n_vars, n_vars), where adj[j, i] = True means j→i.
        """
        T, n_vars = X.shape
        df = pd.DataFrame(X, columns=[str(i) for i in range(n_vars)])
        adj = np.zeros((n_vars, n_vars), dtype=bool)
        for i in range(n_vars):
            validated = self._findcauses(df, str(i))
            for j in validated:
                adj[j, i] = True
        self.adj_matrix = adj
        return adj


# ------ Depthwise TCN modules ------

class Chomp1d(nn.Module):
    """Chops off padding for causal convolution"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class FirstBlock(nn.Module):
    def __init__(self, target, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.conv1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        return self.relu(self.net(x))


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.conv1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        return self.relu(out + x)


class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_inputs)
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # residual in time dimension
        return self.linear(out.transpose(1, 2) + x.transpose(1, 2)).transpose(1, 2)


class DepthwiseNet(nn.Module):
    def __init__(self, target, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super().__init__()
        layers = []
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            pad = (kernel_size - 1) * dilation_size
            if l == 0:
                layers.append(FirstBlock(target, num_inputs, num_inputs,
                                         kernel_size, 1, dilation_size, pad))
            elif l == num_levels - 1:
                layers.append(LastBlock(num_inputs, num_inputs,
                                        kernel_size, 1, dilation_size, pad))
            else:
                layers.append(TemporalBlock(num_inputs, num_inputs,
                                            kernel_size, 1, dilation_size, pad))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ------ ADDSTCN model ------

class ADDSTCN(nn.Module):
    """
    Attention-based Depthwise Separable TCN for one target series.
    """
    def __init__(self, target, input_size, num_levels,
                 kernel_size, cuda, dilation_c):
        super().__init__()
        self.target = target
        self.dwn = DepthwiseNet(target, input_size, num_levels,
                                kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)
        init_att = torch.ones(input_size, 1)
        self.fs_attention = nn.Parameter(init_att)
        if cuda:
            self.cuda()

    def forward(self, x):
        att = F.softmax(self.fs_attention, dim=0)
        y1 = self.dwn(x * att)
        y1 = self.pointwise(y1)
        return y1.transpose(1, 2)
