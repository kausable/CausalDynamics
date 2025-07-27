import warnings

import lingam
import numpy as np
from causallearn.search.FCMBased import lingam as cl_lingam

warnings.filterwarnings("ignore")


class VARLiNGAM:
    """
    VARLiNGAM baseline.

    Reference:
        [1] https://github.com/cdt15/lingam/blob/master/examples/VARLiNGAM.ipynb
    """

    def __init__(self, tau_max: int = 1):
        """Initialize regressor"""
        super(VARLiNGAM, self).__init__()
        self.tau_max = tau_max

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.estimator = lingam.VARLiNGAM(lags=self.tau_max)

        self.estimator.fit(X)
        self.adj_matrix = self.estimator.adjacency_matrices_[self.tau_max]
        self.adj_matrix = np.abs(self.adj_matrix) > 0


class RCD:
    """
    RCD: Repetitive causal discovery of linear non-Gaussian acyclic models with latent confounders.

    Reference:
        [1] International Conference on Artificial Intelligence and Statistics (pp. 735-745). PMLR.
    """

    def __init__(
        self, 
        tau_max: int = 1,
        cor_alpha: float = 0.01,
        ind_alpha: float = 0.01,
        shapiro_alpha: float = 0.01
    ):
        """Initialize regressor"""
        super(RCD, self).__init__()
        self.tau_max = tau_max
        self.cor_alpha = cor_alpha
        self.ind_alpha = ind_alpha
        self.shapiro_alpha = shapiro_alpha

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.estimator = cl_lingam.RCD(cor_alpha=self.cor_alpha, ind_alpha=self.ind_alpha, shapiro_alpha=self.shapiro_alpha)
        self.estimator.fit(X)
        self.adj_matrix = self.estimator.adjacency_matrix_
        self.adj_matrix = np.nan_to_num(self.adj_matrix)

